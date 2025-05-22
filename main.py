from agent import get_agent
from typing import Any, List
from langchain.schema import BaseMessage
from handlers.spinner import spinner
from tools.fact_saver import FactSaver
from longterm_memory import LongTermMemory


def format_chat_history(messages: List[BaseMessage]):
    lines: List[str] = []
    for msg in messages:
        role = "Human" if msg.type == "human" else "AI"
        content = (
            msg.content if isinstance(msg.content, str) # type: ignore
            else str(msg.content) # type: ignore
        )
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def inject_memory_into_input(history: str, user_input: str) -> str:
    return f"{history}\n\nHuman: {user_input}"


def main():
    """
    Main function to run the agent.
    """
    print("Welcome to Agent Cortex! Type 'exit' or 'quit' to stop.")
    agent, memory = get_agent()
    ltm = LongTermMemory()
    fact_saver = FactSaver(ltm)


    while True:

        
        query = input("\n⟁ You: ")
        if query.lower() in ["exit", "quit"]:
            print("Shutting down Cortex")
            break
        stop_spinner = spinner("Thinking...")
        response: None | dict[str, Any] = None
        try: 
            fact_saver.maybe_save_fact(query)

            formatted_history = format_chat_history(memory.chat_memory.messages)
            full_input = inject_memory_into_input(formatted_history, query)

            response = agent.invoke({
                "input": full_input,
                "chat_history": memory,
            })

        except Exception as e:
            print(f"Error: {e}")
        finally:
            stop_spinner()
            if response is not None:
                print(f"\n⚇ Cortex: {response['output']}\n")

if __name__ == "__main__":
    main()