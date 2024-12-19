from gcp_tts_stt import text_to_speech_typecast, recognize_speech_gcp
from llm_handler import generate_response
import sounddevice as sd
import wave
import os
import keyboard

def record_audio(filename, duration=5, samplerate=16000, device_index=1):
    """
    마이크로부터 음성을 녹음하여 WAV 파일로 저장
    :param filename: 저장할 파일 이름
    :param duration: 녹음 시간 (초)
    :param samplerate: 샘플링 레이트 (Hz)
    :param device_index: 마이크 장치 인덱스
    """
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_index)
        sd.wait()  # 녹음 종료 대기

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(2)  # 샘플 폭 (16비트)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())

    except Exception as e:
        print(f"Recording Error: {e}")

def main():
    print("코코넛팀의 졸프 스타트 기술 시연에 오신걸 환영합니다! 말씀해보세요🫡")

    while True:
        # ESC 키로 종료
        if keyboard.is_pressed("esc"):
            print("\n종료중..")
            break

        # 음성 파일 이름 설정
        audio_filename = "input_audio.wav"

        # 마이크로부터 음성 녹음
        record_audio(audio_filename, duration=5, samplerate=16000, device_index=1)

        # Google Cloud Speech-to-Text 실행
        if os.path.exists(audio_filename):
            recognized_text = recognize_speech_gcp(audio_filename)
            if recognized_text:
                try:
                    llm_response = generate_response(recognized_text)
                    if llm_response:
                        print(f"코코넛: {llm_response}")
                        text_to_speech_typecast(llm_response)
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
