from openai import OpenAI
from openai import OpenAIError
import os
from dotenv import load_dotenv
import base64
from flask import Flask, request, jsonify, Response
from PIL import Image
import io
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
app = Flask(__name__)

# 이미지 리사이즈
def resize_image(file, max_size=32):
    image = Image.open(file)
    image = image.convert("RGB")  # RGBA → RGB 변환
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()

# 이미지 base64 인코딩
def encode_image_to_base64(file_storage):
    return base64.b64encode(file_storage.read()).decode("utf-8")

# 프롬프트 정의
PROMPT = """
## 역할
You are a food recognition and nutrition estimation model.

## 작업 지침
When an image is uploaded, follow these steps:

1. Determine whether the image contains food or not.
   - If the image does not contain food, respond only with a valid JSON object like this: (not markdown or code block).
     {
       "is_food": false,
       "message": "This image does not appear to contain food."
     }

2. If food is detected:
   - Identify all visible and distinct food items (assume the meal is for one person).
   - For each food item, include:
     - name: (in Korean)
     - calories: 
     - carb: 
     - confidence: "high", "medium", or "low" — with short explanation
     - portionSize: "small", "medium", or "large"

3. Separate from the items list, determine the **main_item_name**, which is:
   - The most prominent food based on size, central position, or visual dominance.

4. Sum the calories and carbohydrate values from all food items and include as:
   - total_calories
   - total_carb

5. Respond only with a valid JSON object (not markdown or code block).

## 응답 예시
{
  "is_food": true,
  "items": [
    {
      "name": "냉면",
      "calories": 550,
      "carb": 80,
      "confidence": "high - recognizable cold noodle dish with broth",
      "portion_size": "large"
    },
    {
      "name": "삶은 계란",
      "calories": 70,
      "carb": 1,
      "confidence": "high - clearly visible boiled egg",
      "portion_size": "small"
    },
    {
      "name": "소고기",
      "calories": 100,
      "carb": 0,
      "confidence": "high - visible slices of beef",
      "portion_size": "small"
    }
  ],
  "main_item_name": "냉면",
  "total_calories": 720,
  "total_carb": 81
}
"""

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # 1. 이미지 리사이즈 및 base64 인코딩
        resized_bytes = resize_image(request.files["image"])
        base64_str = base64.b64encode(resized_bytes).decode("utf-8")
        full_base64 = f"data:image/jpeg;base64,{base64_str}"

        # 2. GPT-4o 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            n= 1,
            messages=[
                {"role": "system", "content": "You are a helpful food analysis assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": full_base64}
                        }
                    ],
                    "response_format": "json"
                }
            ],
            max_tokens=1000
        )

        print('response 정보:', response)

        # 3. 응답 결과 및 토큰 사용량
        result_raw = response.choices[0].message.content
        usage = response.usage

        try:
          result_json = json.loads(result_raw) # 문자열 JSON을 python 객체 등으로 변환
       
        except (json.JSONDecodeError, TypeError):
            result_json = {"raw_result": result_raw}
        return Response(
            json.dumps({ 
                "result": result_json,
                "tokens_used": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }, ensure_ascii=False),  # 한글 깨짐 방지
            content_type='application/json'
        )

    # 잔액 소진 에러 처리
    except OpenAIError as e:
        if "insufficient_quota" in str(e).lower():
            return Response(
                json.dumps({
                    "error": "잔액이 모두 소진되어 요청이 차단되었습니다.",
                    "insufficient_quota": True
                }, ensure_ascii=False),
                content_type='application/json',
                status=403
            )
        else:
            return Response(
                json.dumps({
                    "error": "GPT 처리 중 에러가 발생했습니다.",
                    "insufficient_quota": False
                }, ensure_ascii=False),
                content_type='application/json',
                status=500
        )


if __name__ == "__main__":
    app.run(debug=True)


# curl -X POST http://localhost:5000/analyze -F image=@images/04.jpg

# 잔액 소진 시 에러 형식
# {
#   "error": {
#     "message": "You exceeded your current quota, please check your plan and billing details.",
#     "type": "insufficient_quota",
#     "code": null,
#     "param": null
#   }
# }

# 앱에서 처리 예시
# if (response.insufficient_quota === true) {
#   hidePhotoUploadSection();
#   showBanner("AI 사용 한도를 초과했습니다. 충전 후 다시 시도해 주세요.");
# }