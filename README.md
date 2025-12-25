# Text2Kanye
Takes a string, and finds the semantically closest Kanye lyric to it (Doesn't take into consideration 2021-2026 songs!). Returns the best match + top-k list + an ambiguity/no-match status.
 
Input:
 
query (String) - the sentence to be processed
 
--rerank (Bool) [Optional] - Rerank or no (default: false)
 
--retrieve-k (Int) [Optional] - How many candidates to retrieve before reranking (default: 20)
 
--top-k (Int) [Optional] How many matches to print (default: 5)
 
--min-score (Float) [Optional] If best similarity is below this, report no match (default: 0.45)
 
--ambiguity-margin (Float) [Optional] If (best - 2nd best) < margin, report ambiguous (default: 0.05)
