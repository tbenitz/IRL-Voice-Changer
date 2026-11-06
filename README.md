# IRL Voice Changer

A Windows control center for **five different real voice-conversion engines** plus a separate experimental **Pocket TTS** microphone-to-speech demo.

The model runners are the upstream implementations, not simulations. Model downloads, Python environments, licenses and device routing remain separate for each engine. See the full installation guide below once the application sources are populated.

**Model engines:** Beatrice v2 (VCClient), RVC (RVC-Project real-time GUI), Seed-VC Realtime, LLVC (stream-simulated file conversion), MeanVC2 (microphone stream).  
**Pocket demo:** microphone speech recognition -> genuine Pocket TTS synthesis. This necessarily waits for spoken words and is not zero-latency voice conversion.

Only use reference voices you own or have permission to use.
