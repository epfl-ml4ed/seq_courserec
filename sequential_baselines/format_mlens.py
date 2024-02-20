import os
import argparse


def save_users(path, name):
    """Save users to .user file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """
    users = []
    with open(os.path.join(path, "users.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(path, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"{user_id}\n")
    return users


def save_movies(path, name):
    """Save movies to .item file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """

    movies = []
    with open(os.path.join(path, "movies.txt"), "r") as f:
        for line in f:
            movies.append(line.strip())

    with open(os.path.join(path, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for i, item_id in enumerate(movies):
            f.write(f"Item_{i}\n")
    movies = [f'Item_{i}' for i, item_id in enumerate(movies)]
    return movies


def save_movie_entity(path, name, movies):
    """Save movie entity to .link file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        movies (list): list of movies
    """
    with open(os.path.join(path, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for i, movie in enumerate(movies):
            f.write(f"Item_{i}\tE_{movie}\n")


def read_movie_genre(path, kg_triplets, movies):
    """Update kg triplets with instructors.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        movies (list): list of movies
    """
    with open(os.path.join(path, "movie_to_genres.txt"), "r") as f:
        for i, line in enumerate(f):
            genres = line.split(' ')
            for genre in genres:
                if genre:
                    kg_triplets.append(
                        [
                            "E_" + movies[int(i)],
                            "genre",
                            "Genre_" + genre,
                        ]
                    )


def read_movie_tags(path, kg_triplets, movies):
    """Update kg triplets to with category.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        movies (list): list of movies
    """
    with open(os.path.join(path, "movie_to_tags.txt"), "r") as f:
        for i, line in enumerate(f):
            tags = line.strip()
            if tags:
                for tag in tags.split():
                    kg_triplets.append(
                        [
                            "E_" + movies[int(i)],
                            "tag",
                            "Tag_" + tag,
                        ]
                    )


def read_movie_acts(path, kg_triplets, movies):
    """Update kg triplets with skills.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        movies (list): list of movies
    """
    with open(os.path.join(path, "movie_to_acts.txt"), "r") as f:
        for i, line in enumerate(f):
            actors = line.strip()
            if actors:
                for act in actors.split():
                    kg_triplets.append(
                        [
                            "E_" + movies[int(i)],
                            "actor",
                            "Person_" + act,
                        ]
                    )

def read_movie_directs(path, kg_triplets, movies):
    """Update kg triplets with skills.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        movies (list): list of movies
    """
    with open(os.path.join(path, "movie_to_directs.txt"), "r") as f:
        for i, line in enumerate(f):
            directors = line.strip()
            if directors:
                for direct in directors.split():
                    kg_triplets.append(
                        [
                            "E_" + movies[int(i)],
                            "director",
                            "Person_" + direct,
                        ]
                    )

def save_kg_triplets(kg_triplets, path, name):
    """Save kg_triplets to file.

    Args:
        kg_triplets (list): list of triplets as a tuple (head_id, relation_id, tail_id)
        path (str): path to save the file
    """
    with open(os.path.join(path, name + ".kg"), "w") as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head_id, relation_id, tail_id in kg_triplets:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")


def save_enrolment(path, name, subset, users, movies, user_timestamp, user_history):
    """Save mlens enrolments to file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        subset (str): name of the subset
        users (list): list of users
        movies (list): list of movies
    """
    enrolments = []
    with open(os.path.join(path, subset + ".txt"), "r") as f:
        for line in f:
            enrolments.append([int(x) for x in line.split()])

    with open(os.path.join(path, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\titem_id_list:token_seq\titem_length:float\n")
        for user, movie in enrolments:
            tmp = user_timestamp.get(user, 0)
            hist = user_history.get(user, [])
            if(len(hist) == 0):
                user_history[user] = ['0']
            hist_str = ' '.join(user_history[user])
            f.write(f"{users[user]}\t{movies[movie]}\t1\t{tmp}\t{hist_str}\t{len(user_history[user])}\n")
            user_timestamp[user] = tmp + 1
            user_history[user] = user_history.get(user, []) + [movies[movie]]


def format_pgpr_coco(datadir):
    """Format PGPR-COCO dataset to recbole format.

    Args:
        datadir (str): path to the dataset
    """
    dataset_name = os.path.basename(os.path.normpath(datadir))
    users = save_users(datadir, dataset_name)
    movies = save_movies(datadir, dataset_name)

    subsets = ['train', "train_target", "test_target"]
    user_timestamp = {}
    user_history = {}
    for subset in subsets:
        save_enrolment(datadir, dataset_name, subset, users, movies, user_timestamp=user_timestamp, user_history=user_history)

    save_movie_entity(datadir, dataset_name, movies)
    kg_triplets = []
    read_movie_tags(datadir, kg_triplets, movies)
    read_movie_genre(datadir, kg_triplets, movies)
    read_movie_acts(datadir, kg_triplets, movies)
    read_movie_directs(datadir, kg_triplets, movies)
    save_kg_triplets(kg_triplets, datadir, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    args = parser.parse_args()

    format_pgpr_coco(args.datadir)
