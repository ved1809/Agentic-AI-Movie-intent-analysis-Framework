from pytube import YouTube
import os

def download_youtube_video(youtube_url, output_dir="downloads"):
    """
    Downloads a YouTube video in the highest resolution available.
    Returns the path to the downloaded video file.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        yt = YouTube(youtube_url)

        print(f"🎬 Title: {yt.title}")
        print(f"📺 Channel: {yt.author}")
        print(f"⏱ Duration: {yt.length // 60} minutes")

        # Get highest resolution progressive stream (includes both audio and video)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        if not stream:
            raise Exception("No compatible stream found!")

        # Download video
        video_path = stream.download(output_path=output_dir)
        print(f"✅ Download complete! File saved at: {video_path}")

        return video_path

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


# Example usage:
if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/watch?v=MyotpcIcR2M"
    video_path = download_youtube_video(youtube_link)
