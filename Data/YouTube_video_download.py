import argparse
import os
import subprocess
import sys
from pathlib import Path
import yt_dlp

def download_video(video_url, output_path="videos"):
    
    # 다운로드 경로 생성
    os.makedirs(output_path, exist_ok=True)
    
    try:  
        # yt-dlp 옵션 설정
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'format': 'best[ext=mp4]',
            'quiet': False,
            'no_cache_dir': True,
            'cachedir': False,
        }
        
        print(f"[✓] 다운로드 시작: {video_url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 영상 정보 가져오기
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'Unknown')
            print(f"[✓] 영상 제목: {title}")
            
            # 다운로드 실행
            ydl.download([video_url])
        
        # 다운로드된 파일 찾기
        downloaded_files = list(Path(output_path).glob(f"{title}.*"))
        if downloaded_files:
            video_path = downloaded_files[0]
            print(f"[✓] 다운로드 완료 → {video_path}")
            return video_path
        else:
            raise FileNotFoundError("다운로드된 파일을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"[✗] 다운로드 실패: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YouTube 영상 다운로드")
    parser.add_argument("url", help="YouTube 동영상 URL")
    parser.add_argument("--output_path", default="videos", help="저장 폴더 (기본값: videos)")
    args = parser.parse_args()
    
    try:
        download_video(args.url, args.output_path)
    except Exception as e:
        print(f"[✗] 프로그램 실행 실패: {e}")
        print("\n[💡] 해결 방법:")
        print("1. 인터넷 연결을 확인하세요")
        print("2. YouTube URL이 올바른지 확인하세요")
        print("3. 영상이 공개되어 있는지 확인하세요")

# python YouTube_video_download.py "YouTube_URL" --output_path "저장경로"