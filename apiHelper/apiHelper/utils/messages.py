class LLMMessageBuilder:
    def __init__(self):
        self.messages = []

    def create_messages_from_text(self, text: str) -> None:
        """
        Creates a new list of messages with one user message from the given text.
        """
        self.messages = [{"role": "user", "content": text}]

    def add_user_message(self, text: str) -> None:
        """
        Adds a user message to the existing messages list.
        """
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, reply: str) -> None:
        """
        Adds an assistant (bot) message to the existing messages list.
        """
        self.messages.append({"role": "assistant", "content": reply})

    def add_system_message(self, instruction: str) -> None:
        """
        Adds a system message to the existing messages list.
        """
        self.messages.append({"role": "system", "content": instruction})

    def clear_messages(self) -> None:
        """
        Clears the messages list.
        """
        self.messages.clear()

    def to_messages(self):
        return self.messages