#!/usr/bin/env python3
"""Test script to verify database connectivity and model creation"""

import sys
import os
from sqlmodel import select
from app.database import get_session, engine
from app.models.chat import Conversation

def test_database_connection():
    print("Testing database connection and models...")

    try:
        # Try to create a session and query (get_session is a generator, so we need to call it differently)
        session_gen = get_session()
        session = next(session_gen)

        try:
            # Try to query conversations table
            statement = select(Conversation).limit(1)
            result = session.exec(statement)
            conversations = result.all()
            print("Successfully queried conversations table.")

            # Try to create a new conversation
            new_conv = Conversation(user_id="test_user")
            session.add(new_conv)
            session.commit()
            session.refresh(new_conv)
            print(f"Successfully created a test conversation with ID: {new_conv.id}")

            # Clean up the test record
            session.delete(new_conv)
            session.commit()
            print("Test record cleaned up successfully")

        finally:
            # Close the session
            session.close()

    except Exception as e:
        print(f"Error testing database: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Database connection and models are working correctly!")
    return True

if __name__ == "__main__":
    test_database_connection()