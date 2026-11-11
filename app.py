"""
مساعدي الخاص - تطبيق محادثة + توليد صور
يعمل كـ Hugging Face Space مجاني بالكامل.

المميزات:
- محادثة نصية بالذاكرة (يتذكر سياق المحادثة)
- توليد صور من وصف نصي
- تحميل الصور الناتجة
- واجهة عربية كاملة
"""

import gradio as gr
from huggingface_hub import InferenceClient
import os

# ---------------------------------------------------------
# الإعدادات: غيّر أسماء الموديلات هنا إذا تبي موديل مختلف
# ---------------------------------------------------------
TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # موديل المحادثة (مفتوح، بدون حاجة لموافقة مسبقة)
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # موديل توليد الصور

# التوكن يُقرأ تلقائيًا من Secrets الخاصة بالـ Space (HF_TOKEN)
# أنشئ توكن مجاني من: huggingface.co/settings/tokens وأضفه كـ Secret باسم HF_TOKEN
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

SYSTEM_PROMPT = "أنت مساعد ذكي ومفيد. أجب دائمًا بوضوح وإيجاز، وبنفس لغة السؤال."


# ---------------------------------------------------------
# المحادثة النصية
# ---------------------------------------------------------
def chat_fn(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat_completion(
            messages=messages,
            model=TEXT_MODEL,
            max_tokens=700,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ تعذّر الوصول للموديل: {e}\n\nتأكد من إضافة HF_TOKEN في إعدادات الـ Space (Settings → Secrets)."


# ---------------------------------------------------------
# توليد الصور
# ---------------------------------------------------------
def generate_image(prompt):
    if not prompt or not prompt.strip():
        return None, "اكتب وصفًا للصورة أولًا."
    try:
        image = client.text_to_image(prompt, model=IMAGE_MODEL)
        return image, ""
    except Exception as e:
        return None, f"⚠️ تعذّر توليد الصورة: {e}"


# ---------------------------------------------------------
# الواجهة
# ---------------------------------------------------------
with gr.Blocks(title="مساعدي الخاص", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("## 🤖 مساعدي الخاص\nمحادثة وتوليد صور — بدون اشتراك، بدون حد يومي.")

    with gr.Tab("💬 محادثة"):
        gr.ChatInterface(
            fn=chat_fn,
            examples=["اشرح لي الذكاء الاصطناعي ببساطة", "اكتب لي بريد إلكتروني رسمي"],
        )

    with gr.Tab("🖼️ توليد صور"):
        with gr.Row():
            with gr.Column():
                img_prompt = gr.Textbox(
                    label="صف الصورة اللي تريدها",
                    placeholder="مثال: منظر طبيعي لجبال عند الغروب، ألوان دافئة",
                    lines=3,
                )
                gen_btn = gr.Button("توليد الصورة", variant="primary")
                err_box = gr.Markdown()
            with gr.Column():
                img_output = gr.Image(label="النتيجة")

        gen_btn.click(fn=generate_image, inputs=img_prompt, outputs=[img_output, err_box])

if __name__ == "__main__":
    demo.launch()
