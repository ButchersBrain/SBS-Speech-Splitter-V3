from .speaker_separator_node import SpeakerSeparatorNode

NODE_CLASS_MAPPINGS = {
    "SpeakerSeparator": SpeakerSeparatorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerSeparator": "Storybook Speech Splitter"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
