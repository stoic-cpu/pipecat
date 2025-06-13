#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.llm_vertex import (
    GoogleVertexLLMService,
)

from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    GeminiMultimodalModalities,
    InputParams,
)
from pipecat.services.gemini_multimodal_live.gemini_vertex import (
    GoogleVertexMultimodalLiveLLMService,
)

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    ##### temp using this as test bed for gemini/vertex refactor #####
    ##### 1. vanilla google llm
    # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
    # # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-live-preview-04-09") ## can't use this model here

    # #### 2. google llm on vertex ai
    # llm = GoogleVertexLLMService(
    #     credentials=os.getenv("GOOGLE_TEST_CREDENTIALS"),
    #     params=GoogleVertexLLMService.InputParams(
    #         project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
    #     ),
    #     # model="google/gemini-2.0-flash-live-preview-04-09", ## can't use this model here
    # )

    # ##### 3. vanilla live llm
    # llm = GeminiMultimodalLiveLLMService(
    #     api_key=os.getenv("GOOGLE_API_KEY"),
    #     # params=InputParams(modalities=GeminiMultimodalModalities.AUDIO),
    #     params=InputParams(modalities=GeminiMultimodalModalities.TEXT),
    #     # model="models/gemini-2.0-flash-live-001"
    #     # model="gemini-2.0-flash-live-preview-04-09", ## can't use this model here
    #     # model="models/gemini-2.0-flash-live-preview-04-09"
    # )

    #### 4. live llm on vertex ai
    llm = GoogleVertexMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        params=GoogleVertexMultimodalLiveLLMService.InputParams(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
            # modalities="TEXT", 
            # modalities=GeminiMultimodalModalities.TEXT, #ug, figure out why this isn't a string later
        ),
        # model="models/gemini-2.0-flash-live-001"
        # model="gemini-2.0-flash-live-preview-04-09"
        model="models/gemini-2.0-flash-live-preview-04-09"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    async def handle_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count == 1:
            # First attempt: Add a gentle prompt to the conversation
            messages.append(
                {
                    "role": "system",
                    "content": "The user has been quiet. Politely and briefly ask if they're still there.",
                }
            )
            print(f"_____17-detect-user-idle.py * handle_user_idle:::::::::::::::")
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        elif retry_count == 2:
            # Second attempt: More direct prompt
            messages.append(
                {
                    "role": "system",
                    "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
                }
            )
            print(f"_____17-detect-user-idle.py * retry:::::::::::::")
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        else:
            # Third attempt: End the conversation
            await user_idle.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await task.queue_frame(EndFrame())
            return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=5.0)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            # stt,
            # user_idle,  # Idle user check-in
            context_aggregator.user(),
            llm,  # LLM
            # tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
