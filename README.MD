# CLIPAway: Harmonizing Focused Embeddings for Removing Objects via Diffusion Models

[Paper](https://arxiv.org/abs/2406.09368) | [Project Website](https://yigitekin.github.io/CLIPAway/) | [Demo](https://huggingface.co/spaces/yigitekin/CLIPAway) | [BibTeX](#bibtex)

![CLIPAway Teaser](./assets/teaser.png)

**CLIPAway: Harmonizing Focused Embeddings for Removing Objects via Diffusion Models**

Yiğit Ekin, Ahmet Burak Yildirim, Erdem Eren Çağlar, Aykut Erdem, Erkut Erdem, Aysegul Dundar

### **Abstract:** 

> Advanced image editing techniques, particularly inpainting, are essential for seamlessly removing unwanted elements while preserving visual integrity. Traditional GAN-based methods have achieved notable success, but recent advancements in diffusion models have produced superior results due to their training on large-scale datasets, enabling the generation of remarkably realistic inpainted images.Despite their strengths, diffusion models often struggle with object removal tasks without explicit guidance, leading to unintended hallucinations of the removed object. To address this issue, we introduce CLIPAway, a novel approach leveraging CLIP embeddings to focus on background regions while excluding foreground elements. CLIPAway enhances inpainting accuracy and quality by identifying embeddings that prioritize the background, thus achieving seamless object removal. Unlike other methods that rely on specialized training datasets or costly manual annotations, CLIPAway provides a flexible, plug-and-play solution compatible with various diffusion-based inpainting techniques. 

### News
- Training and inference codes are released. (22.06.2024)

### Cloning the Repository

```bash
git clone git@github.com:YigitEkin/CLIPAway.git
cd CLIPAway
```

### Source Code Downloads

Our model uses pretrained Alpha-CLIP networks. Make sure that you are at the root directory of the repository and then clone the Alpha-CLIP repository:
To clone the source code of Alpha-CLIP, use:

```bash
git clone https://github.com/SunzeY/AlphaCLIP.git
```
### Environment Setup
Clone Alpha-CLIP and ensure it's in the repository's root directory. Please refer to the [Source Code Downloads](#source-code-downloads). After that, environment setup process can be started.
[Anaconda](https://www.anaconda.com/) is recommended to install the required dependencies. These dependencies are specified in the conda environment named `clipaway`, which can be created and activated as follows:

```bash
conda env create -f environment.yaml
conda activate clipaway
```
### Pretrained Models 
To download the pretrained models for Alpha-CLIP, IP-Adapter, and our MLP projection network use the script that we provide:

```bash
./download_pretrained_models.sh
```

Alternatively, the pretrained models can be downloaded manually as follows:
****NOTE:**** If you are executing the following commands manually, please make sure that you are in the root directory of the repository.
```bash
mkdir ckpts ckpts/AlphaCLIP ckpts/IPAdapter ckpts/CLIPAway
cd ckpts/AlphaCLIP
gdown 1JfzOTvjf0tqBtKWwpBJtjYxdHi-06dbk
cd ../IPAdapter
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
mkdir image_encoder && cd image_encoder
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json
cd ../../CLIPAway
gdown 1lFHAT2dF5GVRJLxkF1039D53gixHXaTx
cd ../../
```
It is important to note that these scripts will download the pretrained models that we have used in our experiments. However, other pretrained models can be used according to users' preferences.

### Dataset Preparation
For training the **MLP projection network**, we provide a dataset class which can be edited according to the dataset of choice. The dataset class is located in `dataset/dataset.py`.

For training and validation datasets the expected file structure is as follows:

```
root_path
├── image1.jpg
├── image2.jpg
└── ...
```
As masks are static full masks, they are not required to be in the dataset folder. Only the images are required.

For test dataset, the expected file structure is as follows:

```
root_path
├── images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks
    ├── image1.png
    ├── image2.png
    └── ...
```
The masks should be in the same folder with the same name as the images. The masks should be in .png format.

**Note:** We have trained our MLP projection network on [COCO 2017 Training Dataset](http://images.cocodataset.org/zips/train2017.zip). for calculating the validation loss, we have used [COCO 2017 Test Dataset](http://images.cocodataset.org/zips/test2017.zip).
The dataset can be downloaded from the provided link or by using the following commands:

```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
rm train2017.zip test2017.zip
```
Make sure that you update the configuration file with the correct paths to the training and validation datasets.

### Training

We provide a code for training the MLP projection network. The training code can be run using the following command:

```bash
python train.py --config <path_to_training_config>
```
It is important to note that the training code uses the training configuration file to set the parameters of the training process. The configuration file should be edited according to the users' preferences.

The description of the parameters in the training configuration file is as follows:

| Args                          | Description                                                        |
| :---------------------------- | :----------------------------------------------------------------- |
| --data_path                   | Path to the training data directory, where the training images are stored. |
| --val_path                    | Path to the validation data directory, used for evaluating the model during training. |
| --train_batch_size            | Batch size for training, which determines the number of samples processed before the model is updated. |
| --val_batch_size              | Batch size for validation, which determines the number of samples processed during evaluation. |
| --lr                          | Learning rate for the optimizer, which controls how much to change the model in response to the estimated error each time the model weights are updated. |
| --weight_decay                | Weight decay (L2 regularization) parameter, which helps to prevent overfitting by penalizing large weights. |
| --eval_interval               | Interval (in steps) at which to evaluate the model.               |
| --save_interval               | Interval (in steps) at which to save the model checkpoint.        |
| --save_path                   | Directory path where the model checkpoints will be saved during training. |
| --image_encoder_path          | Path to the pre-trained image encoder (CLIP) model, which is used for encoding the input images into embeddings. |
| --alpha_vision_ckpt_pth       | Path to the AlphaCLIP checkpoint, which contains pre-trained weights for the AlphaCLIP model used in conjunction with CLIPAway. |
| --mlp_projection_layer_ckpt_path | Path to the MLP projection layer checkpoint if applicable. Used for projecting embeddings if a specific layer is required. Set to `null` if not used. |
| --alpha_clip_id               | Identifier for the AlphaCLIP model variant used. Specifies the version and configuration of the AlphaCLIP model. |
| --epochs                      | Number of training epochs, which defines how many times the training process will iterate over the entire dataset. |
| --number_of_hidden_layers     | Number of hidden layers in the MLP architecture. Affects model capacity and complexity. |
| --alpha_clip_embed_dim        | Embedding dimension for the AlphaCLIP model, which defines the size of the feature vectors produced by the AlphaCLIP encoder. |
| --ip_adapter_embed_dim        | Embedding dimension for the IP-Adapter model, which defines the size of the feature vectors expected as input by the IP-Adapter. |
| --device                      | Device used for training, typically set to `"cuda"` for GPU acceleration, which speeds up the training process. |


### Inference
**Note:** Our best performing model is built upon [SD-Inpaint](https://huggingface.co/runwayml/stable-diffusion-inpainting) which downscales the provided mask to dimension of the latent for determining the inpainting region. This downscaling can cause some inconsistencies and result in artifacts. To prevent this, we advise you to dilate your masks before using them in the inference process. We provide a script for dilating the masks which can be run as follows:

```bash
python3 dilate.py --directory <path_to_masks> --kernel-size 5 --iterations 5
```
the description of the arguments are as follows:
| Args | Description
| :--- | :----------
| --directory | Path to the directory containing the masks
| --kernel-size | Size of the kernel for dilation
| --iterations | Number of iterations for dilation

After dilation, the inference code can be run on a directory of images using the following command:
```bash
python3 inference.py --config <path_to_inference_config>
```
It is important to note that the inference code uses the inference configuration file to set the parameters of the inference process. The configuration file should be edited according to the users' preferences.

The description of the parameters in the inference configuration file is as follows:
| Args                          | Description                                                                                             |
| :---------------------------- | :------------------------------------------------------------------------------------------------------ |
| --device                      | Device used for inference, typically set to `"cuda"` for GPU acceleration, which speeds up the process. |
| --root_path                   | Path to the directory containing the images to be inpainted and masks                                            |
| --image_encoder_path          | Path to the pre-trained image encoder (CLIP) model, which is used for encoding the input images into embeddings. |
| --alpha_clip_ckpt_pth       | Path to the AlphaCLIP checkpoint, which contains pre-trained weights for the AlphaCLIP model used in conjunction with CLIPAway. |
| --alpha_clip_id               | Identifier for the AlphaCLIP model variant used. Specifies the version and configuration of the AlphaCLIP model. |
| --ip_adapter_ckpt_pth         | Path to the IP-Adapter checkpoint, which contains pre-trained weights for the IP-Adapter model used in conjunction with CLIPAway. |
| --sd_model_key                | Key for the SD-Inpaint model variant used. Specifies the version and configuration of the SD-Inpaint model. |
| --number_of_hidden_layers     | Number of hidden layers in the MLP architecture. Affects model capacity and complexity. |
| --alpha_clip_embed_dim        | Embedding dimension for the AlphaCLIP model, which defines the size of the feature vectors produced by the AlphaCLIP encoder. |
| --ip_adapter_embed_dim        | Embedding dimension for the IP-Adapter model, which defines the size of the feature vectors expected as input by the IP-Adapter. |
| --mlp_projection_layer_ckpt_path | Path to the MLP projection layer checkpoint if applicable. Used for projecting embeddings if a specific layer is required. Set to `null` if not used. |
| --save_path_prefix            | Prefix for the output directory where the inpainted images will be saved.                                |
| --seed                        | Seed for the random number generator, which ensures reproducibility of the results.                      |
| --scale | scale parameter of ipadapter model. Determines how much focus is put on the image embeddings. expects a value in range [0,1]. |
| --strength | strength parameter which determines how much forward diffusion is applied. expects a value in range [0,1]. |
| --display_focused_embeds | If set to True, the saved outputs will include unconditional image generations of the focused embeddings as well as the projection block. |

### Gradio 
We provide a gradio interface for the inference process. The interface can be run using the following command:

```bash
python3 app.py --config <path_to_inference_config>
```
If you want to get a shareable link for the interface, you can use the following command:

```bash
python3 app.py --config <path_to_inference_config> --share
```

### Acknowledgement

CLIPAway is implemented on top of the [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) paper which heavily relies on the [Diffusers](https://github.com/huggingface/diffusers) repository. In addition, the [AlphaCLIP](https://github.com/SunzeY/AlphaCLIP) is used for obtaining focused embeddings. We would like to thank the authors of these repositories for their contributions.

### BibTeX

```
@misc{ekin2024clipaway,
      title={CLIPAway: Harmonizing Focused Embeddings for Removing Objects via Diffusion Models}, 
      author={Yigit Ekin and Ahmet Burak Yildirim and Erdem Eren Caglar and Aykut Erdem and Erkut Erdem and Aysegul Dundar},
      year={2024},
      eprint={2406.09368},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
```
