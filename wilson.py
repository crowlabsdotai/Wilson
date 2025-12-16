"""
Wilson:
is an AI voice assistant running on PC and laptops, leveraging the power of the CPU. 
It is built using open-source projects and offers a platform for developers to explore and experiment with various AI-driven scenarios:
including voice-to-text, text-to-speech, and language model interactions.

Contact:
For any questions or inquiries, feel free to contact us at:
Email: team@network-lab.pub
"""

from helper import *

init()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(Fore.CYAN + f"Using device: {device}" + Style.RESET_ALL)

audio_buffer = np.zeros(BUFFER_SIZE)
buffer_index = 0

synthesized_audio_queue = deque()

buffer_lock = threading.Lock()

MQTT_BROKER = "broker.hivemq.com"  
MQTT_PORT = 1883
MQTT_TOPIC_TRANSCRIPTIONS = "transcriptions/whisper"
MQTT_TOPIC_LLM_RESPONSES = "responses/llm"
MQTT_TOPIC_TTS_PLAYING = "transcriptions/tts_playing"

def is_redis_running():
    try:
        output = subprocess.check_output(['redis-cli', 'ping'])
        return output.strip() == b'PONG'
    except Exception:
        return False

def start_redis_server():
    if not is_redis_running():
        print(Fore.CYAN + "Starting Redis server..." + Style.RESET_ALL)
        redis_server_process = subprocess.Popen(['nohup', 'redis-server'])
        sleep(1) 
        if is_redis_running():
            print(Fore.GREEN + "Redis server started successfully!" + Style.RESET_ALL)
        else:
            print(Fore.RED + "Failed to start Redis server." + Style.RESET_ALL)
            exit(1)
    else:
        print(Fore.GREEN + "Redis server is already running." + Style.RESET_ALL)

start_redis_server()

redis_client = redis.Redis(host='localhost', port=6379, db=0)

ollama_model = ChatOllama(model="llama3")

template = """
Your name is Wilson and you are a close friend. You aim to provide coherent conversation of less than 20 words. 
Please always answer in English.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=ollama_model,
)

conversation_count = 0 
summary_counts = 10

def initialize_chromadb():
    global collection
    client = chromadb.Client()
    collection = client.create_collection(name="conversations")
    print(Fore.CYAN + "ChromaDB collection 'conversations' initialized." + Style.RESET_ALL)

initialize_chromadb()

DUMP_FILE_PATH = "./embeddings_dump.json"

def save_embeddings_to_dump():
    with open(DUMP_FILE_PATH, 'w') as f:
        dump_data = collection.get()
        json.dump(dump_data, f)

def generate_embeddings(text):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    embedding = response["embedding"]
    return embedding

def update_vector_db_and_dump(embedding, document):
    global collection
    collection.add(
        ids=[str(conversation_count)],
        embeddings=[embedding],
        documents=[document]
    )
    save_embeddings_to_dump()

def play_synthesized_queue():
    if not synthesized_audio_queue and conversation_count > 1:
        print(Fore.YELLOW + "Synthesized audio queue is empty. Nothing to play." + Style.RESET_ALL)
        play_random_audio('asset/audios/gaps')
        #sleep(5)
        current_time_seconds = T.time()
        is_playing_tts = {"status": False, "time": current_time_seconds}
        set_is_playing_tts(is_playing_tts)
        return

    weight = 1.0
    accumulated_audio = []
    final_sample_rate = None

    while synthesized_audio_queue:
        audio_data = synthesized_audio_queue.popleft()
        sample_rate, audio_array = audio_data
        if final_sample_rate is None:
            final_sample_rate = sample_rate
        accumulated_audio.append(audio_array)

    if accumulated_audio:
        combined_audio = np.concatenate(accumulated_audio) / weight
        combined_audio = librosa.effects.time_stretch(combined_audio, rate=1.0)
        #combined_audio = gaussian_filter1d(combined_audio, sigma=1.0) 
        play_audio(final_sample_rate, combined_audio)
        print(Fore.GREEN + "Playing accumulated TTS audio" + Style.RESET_ALL)

    #sleep(10)    
    current_time_seconds = T.time()
    is_playing_tts = {"status": False, "time": current_time_seconds}
    set_is_playing_tts(is_playing_tts)

def audio_callback(indata, frames, time, status):
    global buffer_index
    current_time_seconds = T.time()
    if status:
        print(Fore.YELLOW + str(status) + Style.RESET_ALL)
    
    with buffer_lock:
        audio_data = np.squeeze(indata)
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        if buffer_index + len(audio_data) < BUFFER_SIZE:
            audio_buffer[buffer_index:buffer_index+len(audio_data)] = audio_data
            buffer_index += len(audio_data)
        else:
            remaining_space = BUFFER_SIZE - buffer_index
            audio_buffer[buffer_index:buffer_index+remaining_space] = audio_data[:remaining_space]
            buffer_index = BUFFER_SIZE
            
        if buffer_index >= BUFFER_SIZE:
            transcription_thread = threading.Thread(target=transcribe_audio, args=(current_time_seconds,))
            transcription_thread.start()
            buffer_index = 0

def is_speech(audio_data, threshold=0.002):
    energy = np.sum(audio_data ** 2) / len(audio_data)
    print(Fore.MAGENTA + f"Audio energy: {energy}" + Style.RESET_ALL)  
    return energy > threshold

def transcribe_audio(current_time_seconds):
    global audio_buffer

    with buffer_lock:
        audio_data = np.copy(audio_buffer)
    
    try:
        audio_data = librosa.resample(audio_data, orig_sr=SAMPLE_RATE, target_sr=16000)
        
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=16000)

        here = is_speech(reduced_noise_audio)
        sleep(1)

        try:
            if get_WOKE() and not get_is_playing_tts()["status"] and here: check_sleep_wilson(reduced_noise_audio)
            elif not get_WOKE() and not get_is_playing_tts()["status"] and here: check_wake_wilson(reduced_noise_audio)
        except Exception as e:
            print(Fore.RED + "Error during detecting wakeupwords:" + Style.RESET_ALL, e)                   

        if not get_WOKE():
            return
        
        if not here:
            print(Fore.YELLOW + "No significant speech detected; skipping transcription." + Style.RESET_ALL)
            current_time_seconds = T.time()
            is_playing_tts = {"status": True, "time": current_time_seconds}
            set_is_playing_tts(is_playing_tts)
            play_synthesized_queue()
            return

        reduced_noise_audio = np.asfortranarray(reduced_noise_audio)
        audio_data = torch.from_numpy(reduced_noise_audio).float().to(device)
        print(Fore.BLUE + "Performing transcription on audio data of shape:" + Style.RESET_ALL, audio_data.shape)

        for attempt in range(10):
            try:
                result = transcribe_audio_stt_tts(audio_data.numpy())
                transcription_text = result
                break
            except Exception as e:
                print(Fore.YELLOW + f"Retrying transcription (attempt {attempt + 1}/10):" + Style.RESET_ALL, e)
                sleep_randomly(attempt + 1, 10)

        if transcription_text:
            if not get_is_playing_tts()["status"]:
                print(Fore.GREEN + "Transcription result:" + Style.RESET_ALL, transcription_text)
                redis_client.rpush("transcriptions", transcription_text)
                print(Fore.CYAN + "Stored transcription in Redis" + Style.RESET_ALL)
                mqtt_client.publish(MQTT_TOPIC_TRANSCRIPTIONS, transcription_text)
                print(Fore.CYAN + f"Published transcription to topic '{MQTT_TOPIC_TRANSCRIPTIONS}'" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Transcription result:" + Style.RESET_ALL, transcription_text)
                mqtt_client.publish(MQTT_TOPIC_TTS_PLAYING, transcription_text)
                print(Fore.CYAN + f"Published TTS playing transcription to topic '{MQTT_TOPIC_TTS_PLAYING}'" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "Empty transcription result; not publishing." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + "Error during transcription:" + Style.RESET_ALL, e)

def generate_summary():
    if redis_client.llen("transcriptions") >= summary_counts:
        last_summary_counts_transcriptions = redis_client.lrange("transcriptions", -summary_counts, -1)
        last_summary_counts_responses = redis_client.lrange("llm_responses", -summary_counts, -1)
        
        combined_text = ""
        for i in range(summary_counts):
            combined_text += f"Conversation {i + 1}:\nUser: {last_summary_counts_transcriptions[i].decode('utf-8')}\nAssistant: {last_summary_counts_responses[i].decode('utf-8')}\n\n"
        
        summary_prompt = f"Here is a conversation log of last interactions:\n{combined_text}\nPlease provide a very short summary of these interactions."
        summary = chain.run(input=summary_prompt, history=[])
        
        print(Fore.MAGENTA + "Summary of the last conversations:" + Style.RESET_ALL)
        print(Fore.MAGENTA + summary + Style.RESET_ALL)

        embedding = generate_embeddings(summary)
        update_vector_db_and_dump(embedding, summary)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(Fore.CYAN + "Connected to MQTT Broker!" + Style.RESET_ALL)
    else:
        print(Fore.RED + f"Failed to connect, return code {rc}" + Style.RESET_ALL)

def on_message(client, userdata, msg):
    global conversation_count
    topic = msg.topic
    payload = msg.payload.decode()
    
    if topic == MQTT_TOPIC_TRANSCRIPTIONS:
        print(Fore.CYAN + f"Received transcription from topic '{MQTT_TOPIC_TRANSCRIPTIONS}': {payload}" + Style.RESET_ALL)
        response = chain.run(input=payload, history=[])
        print(Fore.YELLOW + "LLM Response:" + Style.RESET_ALL, response)
        
        redis_client.rpush("llm_responses", response)
        print(Fore.CYAN + "Stored LLM response in Redis" + Style.RESET_ALL)

        mqtt_client.publish(MQTT_TOPIC_LLM_RESPONSES, response)
        print(Fore.CYAN + f"Published LLM response to topic '{MQTT_TOPIC_LLM_RESPONSES}'" + Style.RESET_ALL)

        conversation_count += 1
        if conversation_count % summary_counts == 0:
            generate_summary()
    
    elif topic == MQTT_TOPIC_LLM_RESPONSES:
        print(Fore.CYAN + f"Received LLM response from topic '{MQTT_TOPIC_LLM_RESPONSES}': {payload}" + Style.RESET_ALL)

        try:
            for attempt in range(10):
                try:
                    sample_rate, audio_array = synthesize_voice_stt_tts(payload)
                    synthesized_audio_queue.append((sample_rate, audio_array))
                    print(Fore.GREEN + "Added synthesized audio to queue" + Style.RESET_ALL)
                    break
                except Exception as e:
                    print(Fore.YELLOW + f"Retrying synthesized (attempt {attempt + 1}/10):" + Style.RESET_ALL, e)
                    sleep_randomly(attempt + 1, 10)
        except Exception as e:
            print(Fore.RED + "Error during TTS synthesis or playback:" + Style.RESET_ALL, e)

def on_publish(client, userdata, mid):
    print(Fore.CYAN + f"Message published with ID: {mid}" + Style.RESET_ALL)

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_publish = on_publish

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()
mqtt_client.subscribe(MQTT_TOPIC_TRANSCRIPTIONS)
mqtt_client.subscribe(MQTT_TOPIC_LLM_RESPONSES)

play_random_audio('asset/audios/starters')
sleep(5)
current_time_seconds = T.time()
is_playing_tts = {"status": False, "time": current_time_seconds}
set_is_playing_tts(is_playing_tts)

with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK):
    print(Fore.CYAN + "Wilson listening..." + Style.RESET_ALL)
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print(Fore.RED + "Interrupted" + Style.RESET_ALL)
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
