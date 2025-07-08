# OmniMotion: Human Motion Generation from Expressive Texts

<p align="left">
  <a href=''>
    <img src='https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white'></a>
  <a href=''>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href=''>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white'></a>
</p>

![teaser_image](./static/images/result.png)

If you find our code or paper helpful, please consider **starring** this repository and citing the following:

```
xxx
```

## :postbox: News

📢 **2023-11-29** --- Initialized the webpage and git project.

## :round_pushpin: Getting Started

  
### 1.1 Set Up Conda Environment
  
```sh
conda env create -f environment.yml
conda activate momask-plus
```

#### 🔁 Alternative: Pip Installation
If you encounter issues with Conda, you can install the dependencies using pip:

```sh
pip install -r requirements.txt
```

✅ Tested on Python 3.8.20.

### 1.2 Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
> (For evaluation only.)
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1sHajltuE2xgHh91H9pFpMAYAkHaX9o57?usp=drive_link) to download the models and evaluators mannually.

### 1.3 Download the Datasets

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the dataset to this repository:

```
cp -r ./HumanML3D/ ./data/humanml3d
```

**OmniMotion** - Download the data from [huggingface](https://huggingface.co/datasets/Ericguo5513/OmniMotion), then place it in the following directory:

```
cp -r ./OmniMotion ./data/omnimotion
```

## :rocket: Play with Pre-trained Model

### 2.1 Motion Generation 

To generate motion from your own text prompts, use:

```
python gen_momask_plus.py
```
You can modify the inference configuration (e.g., number of diffusion steps, guidance scale, etc.) in ``config/eval_momaskplus.yaml``.

### 2.2 Evaluation

Run the following scripts for quantitive evaluation:

```sh
python eval_momask_plus_hml.py    # Evaluate on HumanML3D dataset
python eval_momask_plus.py        # Evaluate on OmniMotion dataset
```

### 2.3 Training

There are two main components in MoMask++, a multi-scale residual motion VQVAE and a generative masked Transformer.

#### Multi-scale Motion RVQVAE

```sh
python train_rvq_hml.py           # Train RVQVAE on HumanML3D
python train_rvq.py               # Train RVQVAE on OmniMotion
```

Configuration files:
* ``config/residual_vqvae_hml.yaml`` (for HumanML3D)
* ``config/residual_vqvae.yaml`` (for OmniMotion)

#### Generative Masked Transformer

```sh
python train_momask_plus_hml.py   # Train on HumanML3D
python train_momask_plus.py       # Train on OmniMotion
```

Configuration files:
* ``config/train_momaskplus_hml.yaml`` (for HumanML3D)
* ``config/train_momaskplus.yaml`` (for OmniMotion)
  
#### Global Motion Refinement

We use a separate lightweight root motion regressor to refine the root trajectory. In particular, this regressor is trained given local motion features to predict root linear velocities. During motion generation, we use this regressor to re-predict the resulting root trajectories which effectively reduces sliding feet.

## :clapper: Visualization

All animations were manually rendered in **Blender** using **Bitmoji** characters.  
An example character is available [here](xxx), and we use [this Blender scene](xxxx) for animation rendering.

---

### Retargeting

We recommend using the [Rokoko Blender add-on](https://www.rokoko.com/integrations/blender) (v1.4.1) for seamless motion retargeting.

> ⚠️ Note: All motions in **OmniMotion** use **T-Pose** as the rest pose.

If your character rig is in **A-Pose**, use the ``rest_pose_retarget.py`` to convert between T-Pose and A-Pose rest poses:


## Acknowlegements

We sincerely thank the open-sourcing of these works where our code is based on: 

[MoMask](https://github.com/EricGuo5513/momask-codes), [VAR](https://github.com/FoundationVision/VAR), [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing), [Muse](https://github.com/lucidrains/muse-maskgit-pytorch), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main) and [MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main)

### Misc
Contact guochuan5513@gmail.com for further questions.
