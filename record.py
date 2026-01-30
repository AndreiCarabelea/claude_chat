"""Deprecated server-side microphone recording.

Streamlit deployments (eg Streamlit Community Cloud) run on a server that
typically does *not* have access to a microphone device.

The app now records audio from the *client browser* using
`streamlit-mic-recorder` (see [`chat_app.py`](chat_app.py:1)).

This module is retained only to avoid breaking old imports in historical files.
"""


class AudioRecorder:
    """Placeholder kept for backward compatibility.

    The previous PyAudio-based implementation was removed to keep deployment
    dependencies compatible (PyAudio requires system audio devices and native
    build toolchains).
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "AudioRecorder is no longer supported. "
            "Use browser recording via streamlit-mic-recorder in chat_app.py."
        )
