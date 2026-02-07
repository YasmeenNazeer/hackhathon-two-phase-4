import sys
from app.services.task_service import TaskService, TaskNotFoundError

class TodoMenu:
    def __init__(self, service: TaskService):
        self.service = service

    def display_menu(self):
        print("\n--- TODO CONSOLE (PHASE 1) ---")
        print("1. View Tasks")
        print("2. Add Task")
        print("3. Complete Task")
        print("4. Update Task")
        print("5. Delete Task")
        print("6. Exit")

    def run(self):
        while True:
            self.display_menu()
            choice = input("\nSelect an option: ")

            if choice == "1":
                self.view_tasks()
            elif choice == "2":
                self.add_task()
            elif choice == "3":
                self.complete_task()
            elif choice == "4":
                self.update_task_cli()
            elif choice == "5":
                self.delete_task()
            elif choice == "6":
                print("Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")

    def view_tasks(self):
        tasks = self.service.get_all_tasks()
        if not tasks:
            print("\nNo tasks found.")
            return

        print("\n--- YOUR TASKS ---")
        for task in tasks:
            print(task)

    def add_task(self):
        title = input("Enter task title: ").strip()
        if not title:
            print("Error: Title cannot be empty.")
            return
        description = input("Enter task description (optional): ").strip()
        task = self.service.add_task(title, description)
        print(f"Task added successfully with ID: {task.id}")

    def complete_task(self):
        try:
            task_id = int(input("Enter task ID to mark as completed: "))
            self.service.complete_task(task_id)
            print(f"Task {task_id} marked as completed.")
        except ValueError:
            print("Error: Please enter a valid numeric ID.")
        except TaskNotFoundError as e:
            print(f"Error: {e}")

    def delete_task(self):
        try:
            task_id = int(input("Enter task ID to delete: "))
            self.service.delete_task(task_id)
            print(f"Task {task_id} deleted successfully.")
        except ValueError:
            print("Error: Please enter a valid numeric ID.")
        except TaskNotFoundError as e:
            print(f"Error: {e}")

    def update_task_cli(self):
        try:
            task_id = int(input("Enter task ID to update: "))
            updates = {}
            
            current_task = self.service.get_task_by_id(task_id)
            if not current_task:
                raise TaskNotFoundError(f"Task with ID {task_id} not found.")

            print(f"\n--- Updating Task {task_id} ---")
            print(f"Current Title: {current_task.title}")
            new_title = input(f"Enter new title (leave blank to keep '{current_task.title}'): ").strip()
            if new_title:
                updates["title"] = new_title

            print(f"Current Description: {current_task.description}")
            new_description = input(f"Enter new description (leave blank to keep '{current_task.description}'): ").strip()
            if new_description:
                updates["description"] = new_description

            print(f"Current Status: {current_task.status.value}")
            new_status_input = input(f"Enter new status (pending/completed, leave blank to keep '{current_task.status.value}'): ").strip().lower()
            if new_status_input:
                if new_status_input in ["pending", "completed"]:
                    updates["status"] = new_status_input
                else:
                    print("Invalid status. Status can only be 'pending' or 'completed'. Status not updated.")

            if updates:
                updated_task = self.service.update_task(task_id, **updates)
                print(f"Task {task_id} updated successfully: {updated_task}")
            else:
                print("No updates provided. Task not changed.")

        except ValueError:
            print("Error: Please enter a valid numeric ID.")
        except TaskNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

