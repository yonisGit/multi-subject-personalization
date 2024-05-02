# MIT License

# Copyright (c) 2023 MIT HAN Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import clip
import torch
from torchvision import transforms
import os
from accelerate import Accelerator
from torchvision.transforms import ToTensor
from PIL import Image


class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                 2.0])] +  # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                             # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[:2] +
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return torch.bmm(src_img_features.unsqueeze(1), gen_img_features.unsqueeze(2)).mean()


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate_vis_only(self, gen_samples, src_images):
        sim_samples_to_img = self.img_to_img_similarity(
            src_images, gen_samples)

        return sim_samples_to_img


if __name__ == '__main__':
    accelerator = Accelerator()
    clip_eval = CLIPEvaluator(device=accelerator.device, clip_model="ViT-L/14")
    base_path = '_____________________' 
    base_path2 = '____________'  
    total_similarity_score = 0
    total_similarities_count = 0
    for subject_dir in os.listdir(base_path):
        print(f'{subject_dir} is evaluated...')
        subject_lst = subject_dir.split('_')
        for subject_name in subject_lst:
            total_subject_similarity_score = 0
            total_subject_similarities_count = 0
            gen_subject_dir = base_path + subject_dir
            for subject_gen_image in os.listdir(gen_subject_dir):
                generated_image = Image.open(gen_subject_dir + '/' + subject_gen_image).convert("RGB")
                generated_image_tensor = (ToTensor()(generated_image).unsqueeze(0) * 2.0 - 1.0)
                real_subject_dir = base_path2 + f'real/{subject_name}'
                subject_count = 0
                for subject_real_image in os.listdir(real_subject_dir):
                    real_image = Image.open(real_subject_dir + '/' + subject_real_image).convert("RGB")
                    real_image_tensor = (ToTensor()(real_image).unsqueeze(0) * 2.0 - 1.0)
                    similarity_score = clip_eval.img_to_img_similarity(generated_image_tensor, real_image_tensor)
                    total_similarity_score += similarity_score
                    total_subject_similarity_score += similarity_score
                    total_subject_similarities_count += 1
                    total_similarities_count += 1

                print('#########################################')
                print(
                    f'For subject {subject_dir}, with {subject_name} the average is: {(total_subject_similarity_score / total_subject_similarities_count)}')
                print('#########################################')

    average_similarity = total_similarity_score / total_similarities_count
    print(f'The average similarity is: {average_similarity}')
