from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from transformers import UperNetForSemanticSegmentation
from pytorch3dunet.unet3d.model import get_model
from transformers import SegformerForSemanticSegmentation, SegformerModel
from torch import nn
import segmentation_models_pytorch as smp

class UNETR_MulticlassSegformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dropout = .1
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = UNETR(
            in_channels=1,
            out_channels=32,
            img_size=(16, 512, 512),
        )
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=3,   #switch to 1 for single class
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            3, 3, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            3, 3, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
    
class CNN3D_no_depth(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.dropout3d = nn.Dropout3d(.1)
        # self.dropout2d = nn.Dropout2d(.3)
        self.relu = nn.ReLU()
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 16, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_5 = nn.Conv3d(16, 16, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_6 = nn.Conv3d(16, 16, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_7 = nn.Conv3d(16, 16, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_8 = nn.Conv3d(16, 1, kernel_size=(
            1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        # self.bn1 = nn.BatchNorm3d(16)
        # self.bn2 = nn.BatchNorm3d(16)

    def forward(self, image):
        # output = self.dropout3d(image)
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output)
        output = self.conv3d_5(output)
        output = self.conv3d_6(output)
        output = self.conv3d_7(output)
        output = self.conv3d_8(output)
        output = output.max(axis=2)[0]
        return output
    
class CNN3D_Segformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.dropout3d = nn.Dropout3d(.1)
        # self.dropout2d = nn.Dropout2d(.3)
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        # output = self.dropout3d(image)
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        # output = self.dropout2d(output)
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3D_Upernet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.xy_encoder_2d = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-tiny", num_labels=1, ignore_mismatched_sizes=True)
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits
        return output


class CNN3D_Segformer64(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 64, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=64
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3D_SegformerB1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b1",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3D_SegformerB2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout3d = nn.Dropout3d(.2)
        self.dropout2d = nn.Dropout2d(.2)
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.dropout3d(image)
        output = self.conv3d_1(output)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.dropout2d(output)
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3D_SegformerBIG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout3d = nn.Dropout3d(0.1)
        self.dropout2d = nn.Dropout2d(0.1)
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32,
            # attention_probs_dropout_prob = 0.3,
            # classifier_dropout_prob = 0.3,
            # drop_path_rate = 0.3,
            # hidden_dropout_prob = 0.3,


        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.dropout3d(image)
        output = self.conv3d_1(output)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.dropout2d(output)
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3D_SegformerB4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout3d = nn.Dropout3d(.1)
        self.dropout2d = nn.Dropout2d(.3)
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b4",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32,


        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.dropout3d(image)
        output = self.conv3d_1(output)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.dropout2d(output)
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class CNN3D_Unet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = smp.Unet(
            # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="tu-efficientnetv2_s",
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=32,
            # model output channels (number of classes in your dataset)
            classes=1,
        )

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output)
        return output


class CNN3D_MANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = smp.MAnet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=32,
            # model output channels (number of classes in your dataset)
            classes=1,
        )

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output)
        return output


class CNN3D_EfficientUnetplusplusb5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(
            3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = smp.EfficientUnetPlusPlus(
            # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="timm-efficientnet-b5",
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=32,
            # model output channels (number of classes in your dataset)
            classes=1,
        )

    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]
        output = self.xy_encoder_2d(output)
        return output
from pytorch3dunet.unet3d.model import UNet3D, DoubleConv, AbstractUNet
class Unet25D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = AbstractUNet(1, 1, False, DoubleConv, 8, 'gcr', conv_kernel_size=(1, 3, 3), conv_padding=(1, 1, 1), is_segmentation=False, )
        print(self.model)
    def forward(self, image):
        output = self.model(image).max(axis=2)[0]
        return output
    
class Unet3D_Segformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 16,
                               "f_maps": 8, "num_groups": 4, "is_segmentation": False})
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3",
            num_labels=1,
            id2label={1: "ink"},
            label2id={"ink": 1},
            ignore_mismatched_sizes=True,
            num_channels=16
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.model(image).max(axis=2)[0]
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
    
class basic_unet3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 1,
                               "f_maps": 8, "num_groups": 4, "is_segmentation": False, "conv_padding":(2, 1, 1)})

    def forward(self, image):
        output = self.model(image).max(axis=2)[0]
        return output
    
    
class Unet3D_Segformer_Jumbo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 32,
                               "f_maps": 8, "num_groups": 4, "is_segmentation": False})
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.model(image).max(axis=2)[0]
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
    

from unetr import UNETR
class UNETR_Segformer(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = UNETR(
            in_channels=1,
            out_channels=32,
            img_size=(16, self.cfg.size, self.cfg.size),
            conv_block=True
        )
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output


class UNETR_effnet_v2(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = UNETR(
            in_channels=1,
            out_channels=32,
            img_size=(16, self.cfg.size, self.cfg.size),
            conv_block=True
        )
        self.encoder_2d = smp.UnetPlusPlus(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="tu-tf_efficientnetv2_xl.in21k_ft_in1k",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights=None,
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=32,
                # model output channels (number of classes in your dataset)
                classes=1,
            )
        self.downscaler = nn.Conv2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        
    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output)
        output = self.downscaler(output)
        output = self.upscaler(output)
        return output

class unet_effnet_v2(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)
        model = Unet3D_full3d_shallow(None, 32)
        # layers = torch.load("/home/ryanc/kaggle/working/outputs/vesuvius_3d/pretrained/pretrained_model.pth")
        # keys_to_remove = ['model.final_conv.weight', 'model.final_conv.bias']

        # # Remove specific keys from the state_dict
        # layers = {k: v for k, v in layers.items() if k not in keys_to_remove}
        # model.load_state_dict(layers, strict=False)
        self.encoder = model
        self.encoder_2d = smp.Unet(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="tu-tf_efficientnetv2_m",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights=None,
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=32,
                # model output channels (number of classes in your dataset)
                classes=1,
            )
        # self.downscaler = nn.Conv2d(
        #     1, 1, kernel_size=(4, 4), stride=2, padding=1)
        # self.upscaler = nn.ConvTranspose2d(
        #     1, 1, kernel_size=(4, 4), stride=2, padding=1)
        
    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        # output = self.dropout(output)
        output = self.encoder_2d(output)
        return output
    
class effnet_v2(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = smp.UnetPlusPlus(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="tu-tf_efficientnetv2_xl.in21k_ft_in1k",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=1,
            )
        
    def forward(self, image):
        output = self.encoder_2d(image)
        return output

class effnet_v2_half(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.pre_upscaler = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.encoder_2d = smp.UnetPlusPlus(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="tu-tf_efficientnetv2_xl.in21k_ft_in1k",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=1,
            )
        self.downscaler = nn.Conv2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1)
        
    def forward(self, image):
        output = self.pre_upscaler(image)
        output = self.encoder_2d(output)
        output = self.downscaler(output)
        # output = self.upscaler(output)
        return output
    
    
class resnet18(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = smp.Unet(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="resnet18",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=1,
            )
        
    def forward(self, image):
        output = self.encoder_2d(image)
        return output
import torch
class resnet18_regression(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = smp.Unet(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="resnet18",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=2,
            )
        self.linear_1 = nn.Linear(22, 22)
        self.linear_2 = nn.Linear(22, 22)
        self.linear_3 = nn.Linear(2, 22)
        self.linear_4 = nn.Linear(44, 2)
        self.dropout = nn.Dropout(p = .1)

        
        
    def forward(self, image, history):
        # history = self.linear_1(self.dropout(history))
        # history = self.linear_2(history)
        img_output = self.encoder_2d(image).max(-1)[0].max(-1)[0]
        img_output = self.linear_3(img_output)
        output = self.linear_4(torch.cat([img_output, img_output], dim=1))
        return output
    
class effnetv2_regression(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = smp.Unet(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="tu-tf_efficientnetv2_xl.in21k_ft_in1k",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=64,
            )
        self.linear = nn.Linear(64, 2)

        
        
    def forward(self, image, history):
        # history = self.linear_1(self.dropout(history))
        # history = self.linear_2(history)
        img_output = self.encoder_2d(image).max(-1)[0].max(-1)[0]
        output = self.linear(img_output)
        return output
    
import timm
class effnetv2_m_regression(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True, num_classes=18, in_chans = 1)
        
    def forward(self, image, history):
        # history = self.linear_1(self.dropout(history))
        # history = self.linear_2(history)
        img_output = self.encoder_2d(image)
        return img_output
    
class resnet_long_regression(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = timm.create_model('resnet18', pretrained=True, num_classes=20, in_chans = 1)
        
    def forward(self, image, history):
        # history = self.linear_1(self.dropout(history))
        # history = self.linear_2(history)
        img_output = self.encoder_2d(image)
        return img_output
    
class resnet_short_regression(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = timm.create_model('resnet18', pretrained=True, num_classes=10, in_chans = 1)
        
    def forward(self, image, history):
        # history = self.linear_1(self.dropout(history))
        # history = self.linear_2(history)
        img_output = self.encoder_2d(image)
        return img_output
    

class Unet3D_full3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 1,
                               "f_maps": 4, "num_groups": 4, "is_segmentation": False, "num_levels":6, "final_sigmoid":False})

    def forward(self, volume):
        output = self.model(volume)
        return output
import timm_3d
class timm3d_efficientnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm_3d.create_model(
            'convnextv2_atto',
            pretrained=False,
            num_classes=0,
            global_pool='',
            in_chans = 1
            )
        res = self.model(torch.randn(1, 1, self.cfg.size, self.cfg.size, self.cfg.size))
        print(res.shape)
        self.lin = nn.Linear((res.shape[1] * res.shape[2] * res.shape[3] * res.shape[4]), int((self.cfg.size//2)*self.cfg.forecast_length*2))
    def forward(self, volume):
        bs = volume.shape[0]
        return self.lin(self.model(volume).reshape(bs, -1))
    
class Unet3d_point_regressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 8,
                               "f_maps": 2, "num_groups": 2, "is_segmentation": False, "num_levels":9, "final_sigmoid":False})
        self.volume_output = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.volume_output6 = nn.Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.downscale1 = nn.Conv3d(8, 32, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.downscale2 = nn.Conv3d(32, 64, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.downscale3 = nn.Conv3d(64, 128, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.downscale4 = nn.Conv3d(128, 256, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.downscale5 = nn.Conv3d(256, 512, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.downscale6 = nn.Conv3d(512, (self.cfg.history+self.cfg.forecast_length)*2, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))        

    def forward(self, volume):
        bs = volume.shape[0]
        unet_out = self.model(volume)
        
        volume_output = self.volume_output(unet_out)
        volume_output = self.volume_output6(volume_output)
    
        output = self.downscale1(unet_out)
        output = self.downscale2(output)
        output = self.downscale3(output)
        output = self.downscale4(output)
        output = self.downscale5(output)
        output = self.downscale6(output)
        output = output.mean(-2).mean(-2).transpose(1, 2).reshape(bs, self.cfg.size, (self.cfg.history+self.cfg.forecast_length),2)
        return output, volume_output[:, 0, :, :, :]

class point_regressor_simple(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.downscale1 = nn.Conv3d(1, 2, kernel_size=4, stride=2, padding=1)
        self.downscale2 = nn.Conv3d(2, 4, kernel_size=4, stride=2, padding=1)
        self.downscale3 = nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1)
        self.downscale4 = nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1)
        self.downscale5 = nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1)
        self.lin = nn.Linear(int(32*((self.cfg.size//32)**3)), int((self.cfg.size//2)*self.cfg.forecast_length*2))

    def forward(self, volume):
        bs = volume.shape[0]
        output = self.downscale1(volume)
        output = self.downscale2(output)
        output = self.downscale3(output)
        output = self.downscale4(output)
        output = self.downscale5(output)
        output = self.lin(output.reshape(bs, -1))
        return output
    
class Unet3D_full3d_xxl(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 1,
                               "f_maps": 32, "num_groups": 4, "is_segmentation": False, "num_levels":8})

    def forward(self, volume):
        output = self.model(volume)
        return output
    
class Unet3D_full3d_shallow(nn.Module):
    def __init__(self, cfg, out_channels = 1):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": out_channels,
                               "f_maps": 32, "num_groups": 4, "is_segmentation": False, "num_levels":5})

    def forward(self, volume):
        # print(volume.dtype, volume.max(), volume.min())
        output = self.model(volume)
        return output

class Unet3D_full3d_64(nn.Module):
    def __init__(self, cfg, out_channels = 1):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": out_channels,
                               "f_maps": 64, "num_groups": 4, "is_segmentation": False, "num_levels":7})

    def forward(self, volume):
        output = self.model(volume)
        return output
    
class Unet3D_more_layers(nn.Module):
    def __init__(self, cfg, out_channels = 1):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": out_channels,
                               "f_maps": 4, "num_groups": 4, "is_segmentation": False, "num_levels":7})

    def forward(self, volume):
        output = self.model(volume)
        return output
    
class Unet3D_full3d_deep(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 1,
                               "f_maps": 4, "num_groups": 4, "is_segmentation": False, "num_levels":8})

    def forward(self, volume):
        output = self.model(volume)
        return output
    
class resnet18_3d(nn.Module):
    def __init__(self, cfg, dropout = .2):
        super().__init__()
        self.cfg = cfg
        self.encoder_2d = smp.Unet(
                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_name="resnet18",
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=cfg.size,
                # model output channels (number of classes in your dataset)
                classes=cfg.size,
            )
        
    def forward(self, image):
        image = torch.transpose(image, 2, 4)
        output = self.encoder_2d(image[:, 0, :, :, :])
        output = torch.transpose(output[:, None, :, :, :], 2, 4)

        return output
