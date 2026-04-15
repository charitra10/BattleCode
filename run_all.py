import os
import glob
import subprocess
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

maps = glob.glob('maps/*.map26')

bot1 = 'claude_harraser/main.py'
bot2 = 'codex_8/main.py'

# Dynamically scale to your CPU cores for maximum speed
WORKERS = os.cpu_count() or 4 

stats = {
    "wins_bot1": 0,
    "wins_bot2": 0,
    "other": 0,
    "reasons": defaultdict(int)
}

def run_match(map_path):
    map_name = os.path.basename(map_path)
    
    # Replays disabled for speed
    cmd = ["cambc", "run", bot1, bot2, map_path]
    
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    
    winner = "Unknown"
    reason = "Unknown Reason"
    
    lines = res.stdout.splitlines()
    for i, line in enumerate(lines):
        if "Winner:" in line:
            # Parse the winner's name and the base reason
            match = re.search(r"Winner:\s+(\S+)\s+\((.*?),\s+turn", line)
            if match:
                winner = match.group(1)
                reason = match.group(2)
            else:
                parts = line.split("Winner:")
                if len(parts) > 1:
                    winner = parts[1].strip().split()[0]
            
            # If it's a resource win, parse the stat table below it
            if "Resources" in reason or "tiebreak" in reason.lower():
                # Scan ahead to find the "Titanium" row
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("Titanium"):
                        # The line immediately above "Titanium" contains the bot names (the headers)
                        header_cols = lines[j-1].split()
                        
                        try:
                            winner_col = header_cols.index(winner)
                            loser_col = 1 - winner_col  # The other bot
                            
                            # Parse Axionite row (which is the row after Titanium)
                            axionite_line = lines[j+1]
                            # Extract the primary numbers before the parenthesis e.g., "0 (0 mined)" -> "0"
                            ax_nums = [int(x) for x in re.findall(r'(\d+)\s*\(', axionite_line)]
                            
                            # Parse Titanium row
                            ti_nums = [int(x) for x in re.findall(r'(\d+)\s*\(', lines[j])]
                            
                            # Determine the exact reason by following the tiebreaker logic
                            if ax_nums and ax_nums[winner_col] > ax_nums[loser_col]:
                                reason = "Resources (Axionite)"
                            elif ti_nums and ti_nums[winner_col] > ti_nums[loser_col]:
                                reason = "Resources (Titanium)"
                            else:
                                # Fallback if both ores were magically tied
                                reason = "Resources (Units/Harvesters)"
                                
                        except (ValueError, IndexError):
                            # If parsing fails for any reason, stick to the generic "Resources (tiebreak)"
                            pass
                        
                        break # We found and parsed the table, stop scanning ahead
            
            break # We found the winner line, stop scanning the stdout
            
    return map_name, winner, reason, duration

print(f"Running matches on {len(maps)} maps with {WORKERS} workers (Replays Disabled)...")

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(run_match, m): m for m in maps}
    
    for future in as_completed(futures):
        map_name, winner, reason, duration = future.result()
        
        print(f"[{duration:.1f}s] {map_name}: {winner} ({reason})")
        
        # Safely track wins 
        if 'claude_harraser' in winner:
            stats["wins_bot1"] += 1
        elif 'codex_8' in winner:
            stats["wins_bot2"] += 1
        else:
            stats["other"] += 1
            
        stats["reasons"][reason] += 1

print("\n--- Final Statistics ---")
print(f"Total Maps: {len(maps)}")
print(f"claude_harraser Wins: {stats['wins_bot1']}")
print(f"codex_8 Wins: {stats['wins_bot2']}")
print(f"Other/Ties: {stats['other']}")

print("\n--- Victory Types ---")
for r, count in sorted(stats["reasons"].items(), key=lambda x: x[1], reverse=True):
    print(f"{r}: {count}")
