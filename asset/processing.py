import numpy as np
import soundfile as sf
import scipy.signal as sps
import asyncio
import websockets
import json

# Your API client code
def synthesize_voice_stt_tts(text):
    async def _synthesize():
        uri = "ws://localhost:8888/ws/tts"
        async with websockets.connect(uri, timeout=120) as websocket:
            await websocket.send(text)
            
            samplerate_data = await websocket.recv()
            samplerate_json = json.loads(samplerate_data)
            samplerate = samplerate_json['samplerate']
            
            audio_data = await websocket.recv()
            audio = np.frombuffer(audio_data, dtype=np.float32)
            
            return samplerate, audio

    return asyncio.run(_synthesize())

# Define your text samples
gaps = [
    "So, yeah, just give me a second to figure this out—I'm almost there, I promise. I just need to work through a couple more details and it’ll all make sense soon, trust me.",
    "You know how it is, sometimes you just need a minute to let everything click, right? It’s just about finding that one missing piece that makes it all come together.",
    "Anyway, let me just think about this for a sec; I was going somewhere with that. I’m sure it’ll come to me in just a moment, I just need to retrace my steps.",
    "Hmm, let me see if I can get this sorted; it might take a moment, but I’m on it. It’s a bit tricky, but I know I’ll be able to work it out if I just focus.",
    "Well, you know, sometimes these things take a little time to really nail down. It’s all about getting the pieces aligned, and then it’ll click into place.",
    "Yeah, so, I’m just trying to wrap my head around this—give me a sec, I’ll get there. I just need to think it through for a bit longer and I’ll have it sorted out.",
    "Okay, let me just take a minute to think this through—I’ll be right with you. Once I’ve worked through this, I’m sure I’ll have a clearer path forward.",
    "I mean, this is a bit of a brain teaser, but I’m working on it—just hang tight. Give me a little bit more time to work out the tricky parts, and I’ll be there.",
    "Right, so, let me piece this together in my head; I’m almost there, just a moment. It’s coming together slowly but surely—I just need to connect the dots.",
    "Yeah, just bear with me for a sec while I figure this out—I’m getting close. I’ve almost got it, just need to sort out one last thing and we’ll be set.",
    "I’m thinking this through—it’s not quite there yet, but it’s getting close. I just need to work out this final step and we should be good to go.",
    "This is taking a bit longer than I expected, but don’t worry, I’ll figure it out soon. It’s just a matter of getting everything to line up properly.",
    "Let me just go over it in my head one more time to make sure I’m not missing anything. I want to make sure everything’s sorted before I move forward.",
    "Give me a second, I need to think this through a little more carefully. I’m almost there, I just need to refine a couple of details.",
    "I’m almost there, just trying to figure out this last little bit. Once I’ve got that sorted, I’ll be all set and ready to move on.",
    "Just hang on for a moment—I’m working on it and it’ll be clear in just a sec. I just need to process what’s going on before I can move forward.",
    "Let me see if I can get this figured out—I’m sure it’ll only take another minute. It’s coming together, but I need to double-check a few things.",
    "It’s taking a bit longer than I thought, but I’m still on it—I’m making progress. Just a little more time and everything will be sorted out.",
    "I’m almost there—just trying to connect the last few dots and make sure it all makes sense. It won’t be much longer, I’m almost at the finish line.",
    "Hold on just a bit longer, I’m really close to figuring this out. I just need a few more moments to work through this final part."
]

starters = [
    "Hey, how's it going? What's new with you? Anything exciting happening lately or are you just taking it easy?",
    "So, what’s been keeping you busy lately? Have you been working on anything interesting or just going with the flow?",
    "Hey there! Got anything exciting going on today? Or is it one of those days where you’re just taking things as they come?",
    "Long time no chat! How have you been? What’s been going on in your world since we last spoke?",
    "Hi! I was just thinking about you—what's up? Anything fun or exciting on the horizon for you?",
    "Hey, do you have a minute? I'd love to catch up. It feels like it's been a while since we last talked!",
    "So, how's your day been so far? Anything noteworthy happening, or is it just a typical day?",
    "Hey! Anything fun planned for the weekend? Got any cool events or just relaxing ahead?",
    "What’s on your mind today? Anything interesting? I’m all ears if you want to share what’s been going on.",
    "Hi there! What's the latest in your world? Anything new and exciting happening with you?",
    "Hey, what’s the story today? I’ve been wondering how you’re doing lately.",
    "What’s been up with you? Feels like it’s been a minute since we last caught up.",
    "So, what’s going on in your life these days? Any exciting projects or news?",
    "Hey, I was wondering what you’ve been up to lately! Anything interesting?",
    "Hi! How’s everything going? Got any fun updates or just keeping things chill?",
    "Hey there! It’s always good to chat. How’s your day shaping up so far?",
    "What’s the latest in your world? Got any plans coming up or just taking it easy?",
    "How have you been since the last time we talked? What’s new with you?",
    "Hey! What’s on the agenda for today? Anything fun in store?",
    "What’s going on in your life these days? Anything exciting or just the usual?",
    "Hey, how’s your week been so far? Anything keeping you busy or just chilling?",
    "What’s happening today? Got any interesting plans or just going with the flow?",
    "Hey, I was thinking about you—how’s it going? What’s new in your world?",
    "What’s the latest on your end? Anything fun or interesting going on today?",
    "Hi! How’s everything with you? Anything exciting happening or just the usual?",
    "So, what’s been going on with you lately? I’d love to hear about it!",
    "Hey there! Got anything fun on the horizon for today or just taking it easy?",
    "How’s life been treating you? What’s new and exciting in your world?",
    "Hey, how’s it going? What’s keeping you busy these days?",
    "Hi there! What’s happening in your world today? Anything fun or just chilling?",
    "What’s new with you? Got any interesting projects or just going with the flow?",
    "So, what’s been keeping you busy lately? Anything cool going on?",
    "Hey, what’s the plan for today? Anything fun or just a chill day?",
    "How’s everything been since we last talked? Got any updates for me?",
    "Hey! How’s it going? What’s the latest in your life these days?",
    "So, how have you been? Anything exciting happening lately?",
    "Hi! What’s on the agenda for today? Got anything cool planned?",
    "What’s new with you today? Anything fun in store or just taking it easy?",
    "Hey there! What’s happening today? Anything exciting or just a regular day?"
]

byes = [
    "Alright, I’ll be here if you need anything else. Catch you later! Just reach out when you’re ready, and I’ll be happy to help.",
    "Okay, I’m going to take a little break now. Talk soon! Let me know if you need me again, and I’ll be right here.",
    "I’ll go quiet for a bit—just say the word if you need me! I’m always here when you’re ready to pick things back up.",
    "Sounds good, I’m going to power down for now. See you later! Feel free to reach out if you need anything at all.",
    "Alright, I’ll be here when you’re ready. Take care! I’ll be right here, waiting if you need any help later on.",
    "I’ll be on standby if you need me. Have a great day! I’m just a call away whenever you need anything.",
    "Okay, I’m going into rest mode. Just call me if you need something! I’ll be ready to assist when you need me again.",
    "I’m going to hibernate for a bit. Don’t hesitate to wake me up! I’m always here whenever you’re ready to continue.",
    "Alright, I’m going offline for now. Catch you next time! Don’t hesitate to reach out when you need me again.",
    "I’ll be here when you need me. Going into hibernation mode now! Just give me a shout when you’re ready to go again.",
    "Alright, I’ll be resting now, but just call and I’ll be back! I’m only a moment away when you need me again.",
    "I’ll be quiet for a bit, but you know where to find me. Take care! Just reach out whenever you need me.",
    "Okay, I’m going offline for a bit—catch you later! Let me know when you need me, and I’ll be right here.",
    "I’m going to take a little break, but I’ll be back when you need me. Catch you soon!",
    "I’m logging off for now, but don’t hesitate to get in touch when you need me! See you later.",
    "I’ll be here when you’re ready, just give me a shout! Catch you later, have a great day.",
    "Okay, I’m stepping away for a bit—just call when you need me again! Talk soon!",
    "I’m powering down for now, but feel free to reach out whenever you need me. Catch you later!",
    "Alright, I’ll be here when you’re ready. Have a great day, and talk to you soon!",
    "I’m going into rest mode now, but just call if you need anything. Catch you later!",
    "Okay, I’m going quiet for a bit, but I’m here whenever you’re ready to continue. Talk soon!",
    "I’ll be offline for a while, but just let me know if you need anything! Catch you later!",
    "I’m going to take a little rest now—reach out if you need anything later. Talk soon!",
    "Okay, I’m stepping back for a bit, but I’ll be here when you’re ready. Catch you soon!",
    "I’m going quiet now, but I’ll be right here when you’re ready for me. Take care and talk soon!",
    "Alright, I’m going to log off for a while—just call me when you need me! Catch you later.",
    "I’m powering down for now, but don’t hesitate to reach out. See you later!",
    "I’m going to hibernate for a bit—let me know if you need anything! Catch you soon.",
    "Alright, I’m logging off for now—just give me a call if you need me! Talk later.",
    "I’ll be here when you’re ready—going into rest mode now. See you next time!",
    "Okay, I’m stepping back for now, but don’t hesitate to call when you’re ready. Catch you later!",
    "I’m going quiet for a bit—just give me a shout when you need me again! See you soon.",
    "Alright, I’m taking a break, but I’ll be here when you need me. Talk to you later!",
    "I’m going to power down for a bit—just reach out when you need me! Catch you later.",
    "I’ll be offline for a while, but I’ll be here when you’re ready to continue. Talk soon!",
    "Okay, I’m going into rest mode for now, but just let me know if you need anything! Catch you later.",
    "I’m stepping away for a bit—just call me when you’re ready to continue. See you soon!",
    "Alright, I’m going offline for a while—let me know if you need anything later. Catch you soon!"
]

folder = "audios/"
new_rate = 16000

# Helper function to process and save audio
def process_and_save_audio(texts, folder_name):
    for text in texts:
        # Synthesize the text using the API
        samplerate, audio = synthesize_voice_stt_tts(text)
        
        # Resample data
        number_of_samples = round(len(audio) * float(new_rate) / samplerate)
        audio = sps.resample(audio, number_of_samples)
        
        # Define output path
        output_path = folder + folder_name + "/" + text.replace(" ", "_").replace(",", "").replace("—", "").replace("'", "") + '.wav'
        
        # Save audio file
        sf.write(output_path, audio, new_rate)

# Process and save each category of texts
process_and_save_audio(gaps, "gaps")
process_and_save_audio(starters, "starters")
process_and_save_audio(byes, "byes")