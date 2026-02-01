# Text2Kanye
Takes a string, and finds the semantically closest Kanye lyric to it (Doesn't take into consideration 2021-2026 songs!). Returns the best match + top-k list + an ambiguity/no-match status.

Commands:
  /q            quit
  /help         show commands
  /maxlen N     set max token length (e.g., /maxlen 96)
  /topk N       set printed top-k
  /retr N       set retrieval-k
