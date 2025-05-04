import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_TOKEN_ID = 49190
NUM_IMG_TOKENS = 64
NUM_PATCHES = 8


def calculate_plt_size(attention_layer_num):
    num_layers = attention_layer_num
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    return rows, cols


def infer_grid_from_token_ids(
    input_ids,
    image_start_token=49189,
    image_patch_token=49190,
    global_image_prefix_token=1116,
    newline_token=198,
    patches_per_image=64,
):
    """
    Infers (rows, cols) of subimages before the global image.
    Splits lines on newline_token (198) and identifies blocks in each.
    """
    block_len = 2 + patches_per_image  # start + pos + 64 patches

    # 1. Split at global image marker
    if global_image_prefix_token not in input_ids:
        # TODO: Handle case where global image is not present
        return 0, 0
    split_idx = input_ids.index(global_image_prefix_token)
    subimage_ids = input_ids[:split_idx]

    # 2. Split subimages into lines (based on newline tokens)
    lines = []
    current_line = []
    for token in subimage_ids:
        if token == newline_token:
            if current_line:
                lines.append(current_line)
                current_line = []
        else:
            current_line.append(token)
    if current_line:
        lines.append(current_line)

    # 3. Count valid blocks per line
    block_counts_per_line = []
    for line in lines:
        i = 0
        count = 0
        while i <= len(line) - block_len:
            if (
                line[i] == image_start_token
                and line[i + 2 : i + 2 + patches_per_image]
                == [image_patch_token] * patches_per_image
            ):
                count += 1
                i += block_len
            else:
                i += 1
        if count > 0:
            block_counts_per_line.append(count)

    rows = len(block_counts_per_line)
    cols = max(block_counts_per_line) if block_counts_per_line else 0

    return rows, cols


def get_attention_map(
    layer_output_attention, layer_general_attention, image_token_indexes, rows, cols
):
    # Get attention scores per token
    att = layer_output_attention[0, :, -1, image_token_indexes].mean(dim=0)
    att = att.to(torch.float32).detach().cpu().numpy()

    general_att = layer_general_attention[0, :, -1, image_token_indexes].mean(dim=0)
    general_att = general_att.to(torch.float32).detach().cpu().numpy()

    # Calculate relative attention
    att_map = att / (general_att + 1e-6)  # prevent divide-by-zero

    # Reshape to match expected output format (global view is in the last 64 tokens)
    sub_images_view_attention = att_map[:-64].reshape(rows, cols, 8, 8)
    global_view_image_attention = att_map[-64:].reshape(8, 8)

    # Initialize full heatmap
    layer_attention_heatmap = np.zeros((rows * 8, cols * 8))

    # Fill heatmap with subimage attention
    for row in range(rows):
        for col in range(cols):
            subimage_attention = sub_images_view_attention[row, col]
            y_offset = row * 8
            x_offset = col * 8
            layer_attention_heatmap[
                y_offset : y_offset + 8, x_offset : x_offset + 8
            ] = subimage_attention

    # Add global attention (if applicable)
    if global_view_image_attention.ndim == 2:
        global_view_image_attention = (
            torch.tensor(global_view_image_attention).unsqueeze(0).unsqueeze(0)
        )
        interpolated = F.interpolate(
            global_view_image_attention,
            size=(rows * 8, cols * 8),
            mode="bilinear",
        )
        layer_attention_heatmap += interpolated.squeeze().cpu().numpy()

    return layer_attention_heatmap


def plot_all_attention_maps(
    prompt_attentions, general_attentions, image_token_indexes, rows, cols
):
    fig_rows, fig_cols = calculate_plt_size(len(prompt_attentions))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 2, fig_rows * 1.5))

    for i, ax in enumerate(axes.flatten()):
        if i < len(prompt_attentions):
            layer_attention_map = get_attention_map(
                prompt_attentions[i],
                general_attentions[i],
                image_token_indexes,
                rows,
                cols,
            )

            ax.imshow(layer_attention_map, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Layer {i + 1}", fontsize=4)
            ax.axis("off")
            if i == 26:
                fig_26 = plt.figure(figsize=(8, 6))
                plt.imshow(
                    layer_attention_map,
                    cmap="hot",
                    interpolation="nearest",
                )
                fig_26.savefig("debug_layer_26_att.png")

        else:
            ax.axis("off")
    fig.savefig("debug_attention_map.png")


def rel_attention_smolvlm(
    image, prompt, general_prompt, model, processor, att_layer=11, debug=False
):
    """
    Compute relative attention scores for SmolVLM.

    Args:
        image: PIL image
        prompt: Question prompt
        general_prompt: General description prompt
        model: SmolVLM model
        processor: SmolVLM processor
        att_layer: Attention layer to use (default: 11)

    Returns:
        att_map: Relative attention map
    """
    messages_query = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_query = processor.apply_chat_template(
        messages_query, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_query],
        images=[image],
        padding=True,
        return_tensors="pt",
        return_row_col_info=True,
    ).to(model.device)

    messages_general = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": general_prompt,
                },
            ],
        }
    ]

    text_general = processor.apply_chat_template(
        messages_general, tokenize=False, add_generation_prompt=True
    )

    general_inputs = processor(
        text=[text_general],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Determine grid dimensions
    input_ids = general_inputs["input_ids"][0].tolist()
    rows, cols = infer_grid_from_token_ids(input_ids)

    # Get image token positions
    token_ids = inputs["input_ids"][0]
    image_token_indexes = [i for i, token in enumerate(token_ids) if token == 49190]

    # with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    outputs_attentions = outputs.attentions

    general_outputs_attentions = model(
        **general_inputs, output_attentions=True
    ).attentions

    if debug:
        plot_all_attention_maps(
            outputs_attentions,
            general_outputs_attentions,
            image_token_indexes,
            rows,
            cols,
        )

    layer_attention_map = get_attention_map(
        outputs_attentions[att_layer - 1],
        general_outputs_attentions[att_layer - 1],
        image_token_indexes,
        rows,
        cols,
    )

    return layer_attention_map


def gradient_attention_smolvlm(
    image, prompt, general_prompt, model, processor, att_layer=11, debug=False
):
    """
    Generates an attention map using gradient-weighted attention from LLaVA model.

    This function computes attention maps from the LLaVA model and weights them by their
    gradients with respect to the loss. It focuses on the attention paid to image tokens
    in the final token prediction, highlighting regions relevant to the prompt.

    Args:
        image: Input image to analyze
        prompt: Text prompt for which to generate attention
        general_prompt: General text prompt (not directly used in this function)
        model: LLaVA model instance
        processor: LLaVA processor for preparing inputs

    Returns:
        att_map: A 2D numpy array of shape (NUM_PATCHES, NUM_PATCHES) representing
                the gradient-weighted attention map
    """
    messages_query = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_query = processor.apply_chat_template(
        messages_query, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_query],
        images=[image],
        # padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Determine grid dimensions
    input_ids = inputs["input_ids"][0].tolist()
    rows, cols = infer_grid_from_token_ids(input_ids)

    # Get image token positions
    token_ids = inputs["input_ids"][0]
    image_token_indexes = [
        i for i, token in enumerate(token_ids) if token == IMAGE_TOKEN_ID
    ]
    # pos = inputs["input_ids"][0].tolist().index(IMAGE_TOKEN_ID)

    # Compute loss
    outputs = model(**inputs, output_attentions=True)
    ce_loss = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -ce_loss(zero_logit, true_class)

    # Compute attention and gradients
    attention = outputs.attentions[att_layer - 1]
    grads = torch.autograd.grad(loss, attention, retain_graph=True)
    grad_att = attention * F.relu(grads[0])

    global_image_token_indexes = image_token_indexes[-NUM_IMG_TOKENS:]
    subimages_token_indexes = image_token_indexes[:-NUM_IMG_TOKENS]

    # Compute the attention maps
    att_map_subimages = (
        grad_att[0, :, -1, subimages_token_indexes]
        .mean(dim=0)
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(rows, cols, 8, 8)
    )

    att_map_global_image = (
        grad_att[0, :, -1, global_image_token_indexes]
        .mean(dim=0)
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(8, 8)
    )

    att_map_full = np.zeros((rows * 8, cols * 8))
    for row in range(rows):
        for col in range(cols):
            subimage_attention = att_map_subimages[row, col]
            y_offset = row * 8
            x_offset = col * 8
            att_map_full[y_offset : y_offset + 8, x_offset : x_offset + 8] = (
                subimage_attention
            )
    # Add global attention
    if att_map_global_image.ndim == 2:
        interpolated = F.interpolate(
            torch.tensor(att_map_global_image).unsqueeze(0).unsqueeze(0),
            size=(rows * 8, cols * 8),
            mode="bilinear",
        )
        att_map_full += interpolated.squeeze().cpu().numpy()

    if debug:
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(att_map_full, cmap="hot", interpolation="nearest")
        plt.title(f"Gradient Attention Map for SmolVLM Layer {att_layer}")
        fig.savefig("debug_layer_grad_att.png")

    return att_map_full
