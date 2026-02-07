specs\database\schema.md

# Database Schema

## Tables

### `user` (Managed by Better Auth)
- `id`
- `email`
- `password` (hashed)
- ... (standard Better Auth fields)

### `task`
- `id`: UUID (Primary Key)
- `user_id`: String (Link to Auth user)
- `title`: String
- `description`: Text
- `is_completed`: Boolean (Default: False)
- `created_at`: Timestamp
- `updated_at`: Timestamp

### `conversation`
- `id`: SERIAL (Primary Key)
- `user_id`: TEXT (Foreign Key to `user.id`)
- `created_at`: TIMESTAMP

### `message`
- `id`: SERIAL (Primary Key)
- `conversation_id`: INTEGER (Foreign Key to `conversation.id`)
- `sender`: TEXT ('user' or 'agent')
- `content`: TEXT
- `created_at`: TIMESTAMP
