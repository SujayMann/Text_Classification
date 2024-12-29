from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import streamlit as st
from torchvision.transforms.functional import pil_to_tensor

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def predict(image):
    print(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def main():
    img_file = st.file_uploader(label='Image', type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        st.image(img_file, caption=img_file.name)
        image = Image.open(img_file)
        if st.button('Predict'):
            result = predict(image)
            st.write(f'Predicted Text: {result}')
        
if __name__ == '__main__':
    main()