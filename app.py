import logging
import os
from io import BytesIO

import fitz
import magic
import torch
from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from GOT.model import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import Conversation, SeparatorStyle
from GOT.utils.utils import KeywordsStoppingCriteria, disable_torch_init
from PIL import Image
from transformers import AutoTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

MODEL_NAME = "/app/GOT_weights/"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"

MODIFIED_CONV_MPT_TEMPLATE = Conversation(
    system="""<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.
Focus on finding tables, and do not omit any information.
Some cells and even column headers may be empty.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)


def create_app():
    logger.info("Creating app")
    logger.info("Initialising model assets")
    disable_torch_init()
    model_name = os.path.expanduser(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="cpu",
        use_safetensors=True,
        pad_token_id=151643,
    ).eval()
    model.to(device="cpu", dtype=torch.bfloat16)
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    logger.info("Model assets initialised")
    app = FastAPI()
    app.model = model
    app.tokenizer = tokenizer
    app.image_processor = image_processor
    app.image_processor_high = image_processor_high
    logger.info("App created")
    return app


app = create_app()


def get_model(request: Request):
    return request.app.model


def get_tokenizer(request: Request):
    return request.app.tokenizer


def get_image_processor(request: Request):
    return request.app.image_processor


def get_image_processor_high(request: Request):
    return request.app.image_processor_high


def pdf_to_image_bytes(pdf_bytes: bytes) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    # 300 DPI / 72 (default PDF DPI)
    zoom = 300 / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes()


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    format_results: bool = Query(False, alias="format"),
    tokenizer: AutoTokenizer = Depends(get_tokenizer),
    model: GOTQwenForCausalLM = Depends(get_model),
    image_processor: BlipImageEvalProcessor = Depends(get_image_processor),
    image_processor_high: BlipImageEvalProcessor = Depends(get_image_processor_high),
):
    logger.info("Received request on /generate")
    logger.info("Mode: %s", "format" if format_results else "ocr")

    color = ""
    use_im_start_end = True
    image_token_len = 256

    content = await file.read(2048)
    mime_type = magic.from_buffer(content, mime=True)
    logger.info("File mime type: %s", mime_type)
    await file.seek(0)
    file_content = await file.read()
    image_bytes = file_content
    if mime_type == "application/pdf":
        image_bytes = pdf_to_image_bytes(file_content)
        logger.info("Converted PDF to image bytes")
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    base_text = "OCR with format: " if format_results else "OCR: "
    qs = f"[{color}] {base_text}" if color else base_text
    if use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # uses a custom conversation template
    conv = MODIFIED_CONV_MPT_TEMPLATE.copy()
    # conv_mode = "mpt"
    # conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    logger.info("Prompt:\n%s", prompt)
    inputs = tokenizer([prompt])
    logger.info("Tokenized prompt")

    input_ids = torch.as_tensor(inputs.input_ids).to("cpu")
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    image_1 = image.copy()
    logger.info("Copied image")
    image_tensor = image_processor(image)
    logger.info("Processed image")
    image_tensor_1 = image_processor_high(image_1)
    logger.info("Processed image")

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logger.info("Moving image to CPU")
        images = [
            (
                image_tensor.unsqueeze(0).half().to("cpu"),
                image_tensor_1.unsqueeze(0).half().to("cpu"),
            )
        ]
        logger.info("Generating output")
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria],
        )

    logger.info("Output IDs:\n%s", output_ids)
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    logger.info("Outputs:\n%s", outputs)
    return {"response": outputs}
