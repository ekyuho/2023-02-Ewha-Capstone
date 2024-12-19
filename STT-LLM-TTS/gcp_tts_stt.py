from google.cloud import texttospeech
from google.cloud import speech
from pydub import AudioSegment
from pydub.playback import play
import os

def text_to_speech_typecast(text, filename="output.mp3", voice_name="ko-KR-Wavenet-D", speaking_rate=1.1):
    """
    Google Cloud Text-to-Speech API를 사용하여 텍스트를 음성으로 변환하고 재생
    :param text: 변환할 텍스트
    :param filename: 저장할 파일 이름
    :param voice_name: 사용할 음성 (기본값: ko-KR-Standard-A)
    :param speaking_rate: 음성 속도 (1.0 = 기본값)
    """
    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(filename, "wb") as out:
            out.write(response.audio_content)

        audio = AudioSegment.from_file(filename)
        play(audio)

        os.remove(filename)
    except Exception as e:
        print(f"TTS Error: {e}")

def recognize_speech_gcp(audio_filename):
    """
    Google Cloud Speech-to-Text API를 사용하여 음성 파일을 텍스트로 변환 (최적화)
    :param audio_filename: 입력 음성 파일 경로
    :return: 변환된 텍스트
    """
    try:
        client = speech.SpeechClient()

        with open(audio_filename, "rb") as audio_file:
            content = audio_file.read()

        # 짧은 오디오 처리에 적합한 RecognitionAudio 구성
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            enable_automatic_punctuation=True,  # 자동 구두점 추가로 응답 품질 개선
            model="latest_short",  # 짧은 응답에 최적화된 모델
            max_alternatives=1  # 가능한 결과 중 최선의 응답만 반환
        )

        # Google API 호출
        response = client.recognize(config=config, audio=audio)

        for result in response.results:
            print(f"나: {result.alternatives[0].transcript}")
            return result.alternatives[0].transcript

    except Exception as e:
        print(f"STT Error: {e}")
