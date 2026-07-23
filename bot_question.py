import os
import pandas as pd
from dotenv import load_dotenv
import discord
from discord.ext import commands
from openai import OpenAI
from surprise import Dataset, Reader, SVD

load_dotenv()

# HINT (Blank 1): read the secret Discord token out of the environment (Day 6/8).
# Which os function reads an environment variable, given its name as a string?
# Your .env file's variable is named TOKEN.
DISCORD_TOKEN = os.getenv("TOKEN")

# HINT (Blank 2): same idea, but for the OpenAI key (Day 8).
# Your .env file's variable is named OPENAI_API_KEY.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4.1"

SYSTEM_PROMPT = (
    "You are a helper for a Discord movie recommendation bot. Given a user's "
    "question, reply with ONLY the movie title they are asking about, and "
    "nothing else. If their message isn't about a movie, reply with exactly: "
    "I can only help with movie recommendations. Please use commands like !!add_user, !!recommend, and !!add_rating."
)

MOVIES_FILE = "ml-100k/u.item"
ORIGINAL_RATINGS_FILE = "ml-100k/u.data"
WORKING_RATINGS_FILE = "working_ratings.data"
USERS_FILE = "register_users.data"


def load_movie_titles():
    movies = pd.read_csv(
        MOVIES_FILE, sep="|", encoding="latin-1",
        usecols=[0, 1], names=["movie_id", "title"],

    )
    return dict(zip(movies["title"], movies["movie_id"]))


def load_registered_users():
    if not os.path.exists(USERS_FILE):
        return {}
    users = pd.read_csv(USERS_FILE)
    return dict(zip(users["discord_username"], users["user_id"]))


def save_registered_user(discord_username, user_id):
    new_row = pd.DataFrame([{"discord_username": discord_username, "user_id": user_id}])
    write_header = not os.path.exists(USERS_FILE)
    new_row.to_csv(USERS_FILE, mode="a", header=write_header, index=False)


def ensure_working_ratings_file_exists():
    if not os.path.exists(WORKING_RATINGS_FILE):
        original = pd.read_csv(ORIGINAL_RATINGS_FILE, sep="\t", header=None)
        original.to_csv(WORKING_RATINGS_FILE, sep="\t", header=False, index=False)


def add_rating_to_working_file(user_id, movie_id, rating):
    new_row = pd.DataFrame([[user_id, movie_id, rating, 0]])
    new_row.to_csv(WORKING_RATINGS_FILE, sep="\t", mode="a", header=False, index=False)


def train_model():
    reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1, 5))
    data = Dataset.load_from_file(WORKING_RATINGS_FILE, reader=reader)
    trainset = data.build_full_trainset()
    # HINT (Blank 3): create the recommendation algorithm (Day 5).
    model = SVD()  # >>> BLANK 3 <<<
    # HINT (Blank 4): train it on the trainset (Day 5).
    # Which method do you call on a model to actually teach it?
    model.fit(trainset)  # >>> BLANK 4 <<<
    return model


def find_movie(search_text):
    search_text = search_text.lower()
    for title, movie_id in movie_titles.items():
        if search_text in title.lower():
            return title, movie_id
    return None


ensure_working_ratings_file_exists()
movie_titles = load_movie_titles()
registered_users = load_registered_users()
model = train_model()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print(f"Loaded {len(movie_titles)} movies and {len(registered_users)} registered users.")


@bot.event
async def on_command_error(ctx, error):
    await ctx.send("Something went wrong — check the spelling of your command and make sure it's a correct movie recommendation command! (e.g, !!recommend, !!add_user, !!add_rating)")
    print(error)


# ==============================================================================
#  WORKED EXAMPLE: add_user is written for you, fully. Read through it
#  carefully — the other two commands below follow the same overall shape:
#  check something, handle the problem case, then do the real work.
# ==============================================================================

@bot.command()
async def add_user(ctx):
    discord_username = ctx.author.name

    if discord_username in registered_users:
        await ctx.send(f"You're already registered, {discord_username}!")
        return

    new_user_id = 9999 + len(registered_users)
    registered_users[discord_username] = new_user_id
    save_registered_user(discord_username, new_user_id)
    await ctx.send(f"Welcome, {discord_username}! You're registered as user {new_user_id}.")


"""
    Let a registered user rate a movie, e.g.  !!add_rating Titanic 5

    Example:
      !!add_rating Titanic 5
      -> Got it — you rated Titanic (1997) a 5.0!

    Requirements:
      - Make sure the user is registered first; if not, tell them to
        register with !!add_user, and stop
      - Make sure the rating is between 1 and 5; if not, tell them so, and stop
      - Use find_movie() to look up the title they typed; if nothing
        matches, tell them, and stop
      - Save the new rating with add_rating_to_working_file()
      - Retrain the model immediately, so this new rating counts right away
      - Send a confirmation message showing the movie title and rating
    """
    # YOUR CODE HERE

@bot.command()
async def add_rating(ctx, movie_title: str, rating: float):
    global model

    discord_username = ctx.author.name
    if discord_username not in registered_users:
        await ctx.send("Please register first using !!add_user!")
        return

    if rating <1 or rating > 5:
        await ctx.send("Please provide a rating from 1 to 5! (e.g, !!add_rating Titanic 5)")
        return

    match = find_movie(movie_title)
    if match is None:
        await ctx.send(f"Couldn't find a movie matching '{movie_title}' Invalid movie name! Please provide a valid name!")
        return

    movie_title, movie_id = match
    user_id = registered_users[discord_username]


    add_rating_to_working_file(user_id, movie_id, rating)
    model = train_model()
    await ctx.send(f"Got it! You have rated {movie_title} a {rating}!")



    """
    Answer a natural-language question, e.g.  !!recommend would I like Titanic?

    Example:
      !!recommend would I like Titanic?
      -> Titanic (1997) — Predicted rating: 4.1

    Requirements:
      - Make sure the user is registered first; if not, tell them to
        register with !!add_user, and stop
      - Send SYSTEM_PROMPT and the user's question to OpenAI using
        client.responses.create(), with model=OPENAI_MODEL — the AI's reply
        (response.output_text) should be just a movie title
      - Use find_movie() to match that guess against the real dataset; if
        nothing matches, tell the user, and stop
      - Use model.predict(user_id, movie_id) to get a predicted rating for
        this specific user and this specific movie (both IDs need to be
        strings)
      - Send the result back, showing the movie title and the predicted rating
    """
    # YOUR CODE HERE


@bot.command()
async def recommend(ctx, *, question: str):
    discord_username = ctx.author.name

    if discord_username not in registered_users:
        await ctx.send("Please register first using !!add_user!")
        return
    user_id = registered_users[discord_username]

    response = client.responses.create(
        model = OPENAI_MODEL,
        input = [
            {"role": "system","content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    movie_title_guess = response.output_text
    match = find_movie(movie_title_guess)
    if match is None:
        await ctx.send(f"Sorry, I couldn't find a movie name matching '{movie_title_guess}'! Please provide a valid name!")
        return

    movie_name, movie_id = match

    prediction = model.predict(str(user_id), str(movie_id))
    predicted_rating = int(prediction.est)

    await ctx.send(f"For '{movie_name}', your predicted rating is {predicted_rating}!")




bot.run(DISCORD_TOKEN)