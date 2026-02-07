-- Migration script for chat feature for PostgreSQL
-- Run this in your database console locally or on Neon.tech

CREATE TABLE IF NOT EXISTS "conversation" (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "message" (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES "conversation"(id) ON DELETE CASCADE,
    sender TEXT NOT NULL, -- 'user' or 'agent'
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
