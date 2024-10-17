"""UTCI model"""

import inspect

import torch  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from torch import nn  # pylint: disable=import-error

from utils import denormalize_array, denormalize_multi_dim_array

CONV_2D = {
    "conv": nn.Conv2d,
    "batchnorm": nn.BatchNorm2d,
    "tconv": nn.ConvTranspose2d,
    "stride": 2,
    "outputpadding": 1,
}


def create_convolution(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    input_channels,
    output_channels,
    kernel_size,
    stride,
    layer,
    weight_norm,
    batch_norm=None,
    output_padding=None,
    bias: bool = False,
):
    """creates a convolutional layer with optional batch normalization and weight normalization"""
    output = weight_norm(
        layer(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=True if bias else batch_norm is None,
        )
        if not output_padding
        else layer(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            output_padding=output_padding,
            bias=True if bias else batch_norm is None,
        )
    )
    if batch_norm is None:
        return output

    return nn.Sequential(*[output, batch_norm(output_channels), nn.ReLU(inplace=True)])


class ConvEncoderDecoder(
    nn.Module
):  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Basic UNet model"""

    def __init__(self, args, bias: bool = False):
        super().__init__()

        def weight_norm(x):  # pylint: disable=invalid-name
            return nn.utils.weight_norm(x)

        kernel_size = 3
        layers = CONV_2D
        self.args = args

        self.encoder1 = create_convolution(
            args.input_channels,
            args.dimension,
            kernel_size,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )
        self.encoder2 = create_convolution(
            args.dimension,
            2 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )
        self.encoder3 = create_convolution(
            2 * args.dimension,
            4 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )
        self.encoder4 = create_convolution(
            4 * args.dimension,
            8 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )

        self.decoder1 = create_convolution(
            8 * args.dimension,
            4 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["tconv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            output_padding=layers["outputpadding"],
            bias=bias,
        )
        self.projection1 = create_convolution(
            8 * args.dimension,
            4 * args.dimension,
            1,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )

        self.decoder2 = create_convolution(
            4 * args.dimension,
            2 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["tconv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            output_padding=layers["outputpadding"],
            bias=bias,
        )
        self.projection2 = create_convolution(
            4 * args.dimension,
            2 * args.dimension,
            1,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )

        self.decoder3 = create_convolution(
            2 * args.dimension,
            args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["tconv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            output_padding=layers["outputpadding"],
            bias=bias,
        )
        self.projection3 = create_convolution(
            2 * args.dimension,
            args.dimension,
            1,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            bias=bias,
        )

        self.restoration = create_convolution(
            args.dimension,
            args.output_channels,
            kernel_size,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=None,
            bias=bias,
        )

        if args.global_channels > 0:
            self.fc = nn.Linear(  # pylint: disable=invalid-name
                args.global_channels, 8 * args.dimension
            )

    def forward(
        self, spatials, scalars=None, statistics=None
    ):  # pylint: disable=too-many-locals
        """forward pass of the model"""
        x_e1 = self.encoder1(spatials)
        x_e2 = self.encoder2(x_e1)
        x_e3 = self.encoder3(x_e2)
        x_e4 = self.encoder4(x_e3)

        if scalars is not None:
            scalars_emb = F.relu(self.fc(scalars)).unsqueeze(-1).unsqueeze(-1)
            x_e4 = x_e4 + scalars_emb

        x_d1 = self.decoder1(x_e4)
        if 1 in self.args.skip:
            x_d1 = self.projection1(torch.cat([x_d1, x_e3], dim=1))

        x_d2 = self.decoder2(x_d1)
        if 2 in self.args.skip:
            x_d2 = self.projection2(torch.cat([x_d2, x_e2], dim=1))

        x_d3 = self.decoder3(x_d2)
        if 3 in self.args.skip:
            x_d3 = self.projection3(torch.cat([x_d3, x_e1], dim=1))

        out = self.restoration(x_d3)
        if statistics is not None:
            if isinstance(statistics, tuple):
                mean, std = statistics
                return denormalize_array(out, mean, std)
            means = [l[0] for l in statistics]
            stds = [l[1] for l in statistics]
            return denormalize_multi_dim_array(out, means, stds)
        return out


class UTCIModel(nn.Module):
    """The model is trained with UTCI supervision only"""

    def __init__(self, utci_model: nn.Module, channels_last: bool = False) -> None:
        super().__init__()
        self.utci_model = utci_model
        self.channels_last = channels_last

    @property
    def forward_input_kwargs(self):
        """returns the arguments of the forward method"""
        return [
            arg for arg in inspect.getfullargspec(self.forward).args if arg != "self"
        ]

    def forward(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        spatial: torch.Tensor,
        wind_predictors: torch.Tensor,
        temporal: torch.Tensor,
        mlp_spatial: torch.Tensor,
        statistics: tuple,
    ) -> torch.Tensor:
        """Forward pass of UTCI model

        Args:
            spatial (torch.Tensor): spatial inputs of study area; b x 16 x h x w (without aveg SVFs)
                or b x 21 x h x w (with aveg SVFs);
                order: ['r.DEM', 'r.DSM.GB', 'r.DSM.V', 'r.LCC', 'r.WA', 'r.WH', 'svf', 'svfE',
                'svfEveg', 'svfN', 'svfNveg', 'svfS', 'svfSveg', 'svfW', 'svfWveg', 'svfveg']
                (without aveg SVFs), ['r.DEM', 'r.DSM.GB', 'r.DSM.V', 'r.LCC', 'r.WA', 'r.WH',
                'svf', 'svfE', 'svfEaveg', 'svfEveg', 'svfN', 'svfNaveg', 'svfNveg', 'svfS',
                'svfSaveg', 'svfSveg', 'svfW', 'svfWaveg', 'svfWveg', 'svfaveg', 'svfveg']
                (with aveg SVFs)
            wind_predictors (torch.Tensor): wind predictors; b x 18 x h x w;
                order: ['cl_bl_ht', 'dwd_e', 'dwd_n', 'dwd_s', 'dwd_w', 'horzontal_distance',
                'pc_ew', 'pc_ns', 'sli_e', 'sli_n', 'sli_s', 'sli_w', 'swi_ew', 'swi_ns', 'uwd_e',
                'uwd_n', 'uwd_s', 'uwd_w']
            temporal (torch.Tensor): meteorological inputs; b x 9;
                order: ['Ta', 'Wind', 'Wd', 'Kdown', 'rain', 'RH', 'press', 'ElevationAngle',
                'AzimuthAngle']
            mlp_spatial (torch.Tensor): spatially aggregated inputs; b x 17;
                order: ['Baresoil', 'Buildings', 'DecidiousTrees', 'EvergreenTrees', 'Grass',
                'Paved', 'Water', 'alt', 'fai_b', 'fai_v_d', 'fai_v_e', 'popdens', 'z0_b', 'zH_b',
                'zH_v_d', 'zH_v_e', 'zd_b']
            statistics (tuple): mean & std of UTCI; (mean, std)

        Returns:
            torch.Tensor: UTCI; b x 1 x h x w
        """
        spatial = torch.cat([spatial, wind_predictors], dim=1)
        scalars = torch.cat([temporal, mlp_spatial], dim=1)
        if self.channels_last:
            return self.utci_model(
                spatial.to(memory_format=torch.channels_last),
                scalars,
                statistics=statistics,
            )
        return self.utci_model(spatial, scalars, statistics=statistics)

    @torch.inference_mode()
    def forward_fast(self, spatial_wind_predictors, temporal, mlp_spatial, statistics):
        """slightly faster forward pass that avoids the concatenation of the spatial
        and wind inputs"""
        return self.utci_model(
            spatial_wind_predictors,
            torch.cat([temporal, mlp_spatial], dim=1),
            statistics=statistics,
        )
