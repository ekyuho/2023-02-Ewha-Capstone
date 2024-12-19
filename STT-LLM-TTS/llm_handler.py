import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt):
    """
    OpenAI GPT를 사용하여 사용자 입력에 대한 응답 생성 (항상 반말로 대답하도록 설정)
    :param prompt: 사용자 입력 텍스트
    :return: ChatGPT의 응답
    """
    try:
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 모델 이름
            messages=[
                {"role": "system", "content": "너는 항상 반말로 대답하는 어시스턴트야."},
                {"role": "user", "content": f"사용자가 이렇게 말했어: '{prompt}'. 반말로 대답해."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return "문제가 생겼어. 다시 시도해볼래?"

