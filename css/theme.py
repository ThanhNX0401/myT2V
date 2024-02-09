from __future__ import annotations

from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import os

image = 'bk.png'
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, image)


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary= "#FFFFFF",
            background_fill_secondary= "#dce3e8",
            block_background_fill= "#ECF2F7",
            block_border_color= "#dce3e8",
            block_border_width= "1px",
            block_info_text_color= "#191919",
            block_info_text_size= "*text_sm",
            block_info_text_weight= "400",
            block_label_background_fill= "#ECF2F700",
            block_label_border_color= "#dce3e8",
            block_label_border_width= "1px",
            block_label_margin= "0",
            block_label_padding= "*spacing_sm *spacing_lg",
            block_label_radius= "calc(*radius_lg - 1px) 0 calc(*radius_lg - 1px) 0",
            block_label_right_radius= "0 calc(*radius_lg - 1px) 0 calc(*radius_lg - 1px)",
            block_label_shadow= "*block_shadow",
            block_label_text_color= "#4EACEF",
            block_label_text_size= "*text_sm",
            block_label_text_weight= "400",
            block_padding= "*spacing_xl calc(*spacing_xl + 2px)",
            block_radius= "*radius_lg",
            block_shadow= "#FFFFFF00",
            block_title_background_fill= "#ECF2F700",
            block_title_border_color= "#dce3e8",
            block_title_border_width= "0px",
            block_title_padding= "0",
            block_title_radius= "none",
            block_title_text_color= "#4EACEF",
            block_title_text_size= "*text_md",
            block_title_text_weight= "bold",
            body_background_fill= f"url('file={image_path}') #FFFFFF no-repeat padding-box fixed",
            body_text_color= "#191919",
            body_text_color_subdued= "#636668",
            body_text_size= "*text_md",
            body_text_weight= "400",
            border_color_accent= "#dce3e8",
            border_color_accent_subdued= "#dce3e867",
            border_color_primary= "#dce3e8",
            button_border_width= "*input_border_width",
            button_cancel_background_fill= "#dce3e8",
            button_cancel_background_fill_hover= "#d0d7db",
            button_cancel_border_color= "#191919",
            button_cancel_border_color_hover= "#202020",
            button_cancel_text_color= "#4EACEF",
            button_cancel_text_color_hover= "#0c6ebd",
            button_large_padding= "*spacing_lg calc(2 * *spacing_lg)",
            button_large_radius= "*radius_lg",
            button_large_text_size= "*text_lg",
            button_large_text_weight= "600",
            button_primary_background_fill= "#4EACEF",
            button_primary_background_fill_hover= "#0c6ebd",
            button_primary_border_color= "#191919",
            button_primary_border_color_hover= "#202020",
            button_primary_text_color= "#ECF2F7",
            button_primary_text_color_hover= "#e1eaf0",
            button_secondary_background_fill= "#dce3e8",
            button_secondary_background_fill_hover= "#d0d7db",
            button_secondary_border_color= "#dce3e8",
            button_secondary_border_color_hover= "#d0d7db",
            button_secondary_text_color= "#4EACEF",
            button_secondary_text_color_hover= "#0c6ebd",
            button_shadow= "none",
            button_shadow_active= "none",
            button_shadow_hover= "none",
            button_small_padding= "*spacing_sm calc(2 * *spacing_sm)",
            button_small_radius= "*radius_lg",
            button_small_text_size= "*text_md",
            button_small_text_weight= "400",
            button_transition= "background-color 0.2s ease",
            color_accent= "*primary_500",
            color_accent_soft= "#dce3e8",
            color_accent_soft_dark= "#242424",
            container_radius= "*radius_lg",
            embed_radius= "*radius_lg",

            form_gap_width= "0px",
            input_background_fill= "#dce3e8",
            input_background_fill_focus= "#dce3e8",
            input_background_fill_hover= "#d0d7db",
            input_border_color= "#191919",

            input_border_color_focus= "#191919",

            input_border_color_hover= "#202020",

            input_border_width= "0px",
            input_padding= "*spacing_xl",
            input_placeholder_color= "#19191930",

            input_radius= "*radius_lg",
            input_shadow= "#19191900",
            input_shadow_focus= "#19191900",
            input_text_size= "*text_md",
            input_text_weight= "400",
            layout_gap= "*spacing_xxl",

            loader_color= "#4EACEF",
            panel_background_fill= "#ECF2F7",
            panel_border_color= "#4EACEF",
            panel_border_width= "0",
            prose_header_text_weight= "600",
            prose_text_size= "*text_md",
            prose_text_weight= "400",
            radio_circle= "url(\"data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='8' cy='8' r='3'/%3e%3c/svg%3e\")",
            section_header_text_size= "*text_md",
            section_header_text_weight= "400",
            shadow_drop= "rgba(0,0,0,0.05) 0px 1px 2px 0px",
            shadow_drop_lg= "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            shadow_inset= "rgba(0,0,0,0.05) 0px 2px 4px 0px inset",
            shadow_spread= "#FFFFFF",
            slider_color= "#4EACEF",
            stat_background_fill= "#4EACEF",
        )