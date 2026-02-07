from app.services.task_service import TaskService
from app.cli.menu import TodoMenu

def main():
    service = TaskService()
    menu = TodoMenu(service)
    try:
        menu.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
