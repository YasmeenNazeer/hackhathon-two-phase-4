const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

async function main() {
    let databaseUrl;
    try {
        const envPath = path.join(process.cwd(), '.env');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const match = envContent.match(/DATABASE_URL=(["'])?(.*?)\1?(\r?\n|$)/);
        if (match) {
            databaseUrl = match[2];
        }
    } catch (err) {
        console.error("Could not read .env file", err);
    }

    if (!databaseUrl) {
        console.error("DATABASE_URL not found in .env");
        process.exit(1);
    }

    const pool = new Pool({
        connectionString: databaseUrl,
        ssl: { rejectUnauthorized: false }
    });

    const sql = `
    ALTER TABLE "session" ADD COLUMN IF NOT EXISTS "ipAddress" TEXT;
    ALTER TABLE "session" ADD COLUMN IF NOT EXISTS "userAgent" TEXT;
    `;

    try {
        console.log("Checking and adding missing columns to session table...");
        await pool.query(sql);
        console.log("Columns verified/added successfully!");
    } catch (err) {
        console.error("Error updating table:", err);
        process.exit(1);
    } finally {
        await pool.end();
    }
}

main();
