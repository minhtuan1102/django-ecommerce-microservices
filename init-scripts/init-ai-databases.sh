#!/bin/bash
# Database initialization script for AI services
# Creates databases for behavior analysis and chatbot services

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AI Services Database Initialization...${NC}"

# Function to execute SQL
execute_sql() {
    local db_name=$1
    local sql=$2
    
    psql -U bookstore_user -d postgres -c "$sql"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Database '$db_name' initialized${NC}"
    else
        echo -e "${GREEN}✓ Database '$db_name' already exists or initialization complete${NC}"
    fi
}

# Create behavior analysis database
echo -e "${BLUE}Creating behavior_db...${NC}"
execute_sql "behavior_db" "CREATE DATABASE behavior_db OWNER bookstore_user;"

# Create chatbot database
echo -e "${BLUE}Creating chatbot_db...${NC}"
execute_sql "chatbot_db" "CREATE DATABASE chatbot_db OWNER bookstore_user;"

# Grant privileges
echo -e "${BLUE}Setting up privileges...${NC}"
psql -U bookstore_user -d postgres <<-EOSQL
    GRANT CONNECT ON DATABASE behavior_db TO bookstore_user;
    GRANT CONNECT ON DATABASE chatbot_db TO bookstore_user;
    
    -- Connect to behavior_db and grant schema privileges
    \c behavior_db
    GRANT ALL PRIVILEGES ON SCHEMA public TO bookstore_user;
    
    -- Connect to chatbot_db and grant schema privileges
    \c chatbot_db
    GRANT ALL PRIVILEGES ON SCHEMA public TO bookstore_user;
EOSQL

echo -e "${GREEN}AI Services Database Initialization Complete!${NC}"
echo ""
echo "Databases created:"
echo "  - behavior_db (for Behavior Analysis Service)"
echo "  - chatbot_db (for Consulting Chatbot Service)"
echo ""
echo "Tables will be created automatically when services start."
