from pybaseball import statcast, batting_stats

df = batting_stats(2019, league='all')
df.head()