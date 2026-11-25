import pandas as pd
import os

custom_data = {
    "PauseMusic": [
        "pause", "pause the music", "pause song", "stop playing", "hold on pause the track",
        "can you pause the music", "pause playback", "freeze the music", "stop the song",
        "pause this", "pause audio", "put the music on pause", "pause right now",
        "halt playback", "pause my music", "stop music playback", "pause current song",
        "pause the track please", "temporarily stop the music", "pause what is playing",
        "hit pause", "press pause", "pause yt music", "stop the audio", "pause tune",
        "could you pause", "pause for a second", "stop playing this", "music pause",
        "pause the playlist", "pause album", "suspend playback", "stop song right now",
        "pause it", "pause this song", "mute and pause", "stop the track", "pause player",
        "pause current track", "hold the music", "pause playback now", "stop playing audio",
        "pause the stream", "can you stop the music", "please pause", "pause session",
        "stop the playlist", "pause this track", "take a break pause music", "pause background music",
        "pause music right now", "hold playback", "pause the audio stream", "stop playing music",
        "can you pause this", "pause the current tune", "freeze playback", "pause yt",
        "temporarily halt music", "pause the radio"
    ],
    "ResumeMusic": [
        "resume", "resume music", "continue playing", "unpause", "play again",
        "resume the song", "keep playing", "start the music again", "unpause the track",
        "continue playback", "resume my music", "play from where it stopped", "unpause audio",
        "continue the song", "resume playback", "start playing again", "unpause please",
        "carry on playing", "resume the playlist", "continue the track", "hit play",
        "press play", "resume current song", "keep the music going", "restart playback",
        "unpause my music", "continue audio", "resume tune", "play the music again",
        "resume what was playing", "unpause it", "continue this song", "resume player",
        "start music back up", "unpause playback", "go back to playing", "resume stream",
        "can you resume the music", "please unpause", "keep playing the playlist", "resume audio",
        "continue playing music", "unpause this track", "play again please", "resume session",
        "bring the music back", "unpause song", "continue current track", "resume right now", "start playing",
        "resume the audio", "continue playing the song", "play the music back", "unpause the stream",
        "resume yt music", "keep the track playing", "start playback again", "unpause current song",
        "go ahead and resume", "resume playing"
    ],
    "NextTrack": [
        "next", "next song", "skip", "skip this song", "play the next track",
        "skip track", "next track please", "go to the next song", "skip this",
        "forward to next song", "play next", "skip ahead", "next tune",
        "change the song", "skip this one", "move to the next track", "next in queue",
        "skip current song", "next music", "pass this song", "skip forward",
        "next track", "jump to next song", "skip the track", "play next song",
        "can you skip this", "i dont like this song skip", "next on the playlist", "skip to next",
        "go next", "next audio", "skip it", "next song please", "advance track",
        "skip this tune", "move on to next song", "play the following song", "skip playlist track",
        "hit next", "press next", "next please", "skip current track", "change track",
        "go to next", "skip song now", "next one", "forward track", "skip this music",
        "play next tune", "jump ahead one song",
        "skip to the next track", "move forward one song", "next song in queue", "skip this track please",
        "play the next audio", "go to next track", "next music track", "skip ahead a song",
        "switch to next song", "advance to next track"
    ],
    "PreviousTrack": [
        "previous", "previous song", "go back", "play the previous track", "last song",
        "previous track", "go back a song", "play last song", "back to previous track",
        "rewind to last song", "previous tune", "play the song before this", "go back one track",
        "back track", "return to previous song", "play previous", "last track please",
        "go to previous song", "back to the last song", "previous song please", "step back a track",
        "play that last song again", "previous in queue", "go back to the previous track", "back one song",
        "previous audio", "move back a song", "play earlier song", "hit previous",
        "press previous", "previous please", "go back to last track", "return to last song",
        "previous track now", "back to earlier track", "play the prior song", "rewind one track",
        "go backwards a song", "last tune", "previous music", "back a track",
        "play song before", "jump back one song", "previous one", "go back please",
        "back to previous", "play last track", "step back one song", "prior track", "go to last song",
        "replay the previous song", "go back to the last tune", "play the track before this",
        "previous song in queue", "rewind to previous track", "back to the prior song",
        "play previous music", "switch to previous track", "move back to last song", "go to previous"
    ],
    "VolumeControl": [
        "volume up", "volume down", "turn it up", "turn it down", "louder",
        "quieter", "increase volume", "decrease volume", "raise the volume", "lower the volume",
        "mute", "unmute", "set volume to 50 percent", "turn the music up", "turn the music down",
        "make it louder", "make it quieter", "volume 80", "volume 20", "max volume",
        "minimum volume", "turn down the sound", "turn up the sound", "mute the music",
        "unmute the audio", "boost the volume", "reduce volume", "volume to maximum",
        "volume to half", "lower audio", "raise audio", "crank it up", "soften the music",
        "set volume to 30", "set volume to 70", "volume up please", "volume down please",
        "can you turn it up", "can you turn it down", "mute playback", "unmute playback",
        "turn volume up", "turn volume down", "adjust volume to 60", "sound up",
        "sound down", "increase the sound", "decrease the sound", "silence the music", "restore volume",
        "turn up the music volume", "turn down the music volume", "mute the sound", "unmute the sound",
        "drop the volume", "bump up the volume", "set volume to 40 percent", "lower the sound please",
        "raise the volume please", "quiet down the music", "make the music louder"
    ]
}

rows = []
for label, phrases in custom_data.items():
    for text in phrases:
        rows.append({"text": text.lower().strip(), "label": label})

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("data/custom_playback.csv", index=False)
print(f"Saved {len(df)} custom playback commands to data/custom_playback.csv")