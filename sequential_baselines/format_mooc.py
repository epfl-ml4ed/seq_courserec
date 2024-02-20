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


def save_courses(path, name):
    """Save courses to .item file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """

    courses = []
    with open(os.path.join(path, "courses.txt"), "r") as f:
        for line in f:
            courses.append(line.strip())

    with open(os.path.join(path, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for i, item_id in enumerate(courses):
            f.write(f"Item_{i}\n")
    courses = [f'Item_{i}' for i, item_id in enumerate(courses)]
    return courses


def save_course_entity(path, name, courses):
    """Save course entity to .link file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        courses (list): list of courses
    """
    with open(os.path.join(path, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for i, course in enumerate(courses):
            f.write(f"Item_{i}\tE_{course}\n")


def read_course_concept(path, kg_triplets, courses):
    """Update kg triplets with instructors.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_concepts.txt"), "r") as f:
        for i, line in enumerate(f):
            concepts = line.split(' ')
            for concept in concepts:
                if concept:
                    kg_triplets.append(
                        [
                            "E_" + courses[int(i)],
                            "concept",
                            "Concept_" + concept,
                        ]
                    )

def read_course_teachers(path, kg_triplets, courses):
    """Update kg triplets with skills.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_teachers.txt"), "r") as f:
        for i, line in enumerate(f):
            teachers = line.strip()
            if teachers:
                for teacher in teachers.split():
                    kg_triplets.append(
                        [
                            "E_" + courses[int(i)],
                            "teacher",
                            "Teacher_" + teacher,
                        ]
                    )

def read_course_schools(path, kg_triplets, courses):
    """Update kg triplets with skills.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_school.txt"), "r") as f:
        for i, line in enumerate(f):
            schools = line.strip()
            if schools:
                for school in schools.split():
                    kg_triplets.append(
                        [
                            "E_" + courses[int(i)],
                            "school",
                            "School_" + school,
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


def save_enrolment(path, name, subset, users, courses, user_timestamp, user_history):
    """Save mooc enrolments to file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        subset (str): name of the subset
        users (list): list of users
        courses (list): list of courses
    """
    enrolments = []
    with open(os.path.join(path, subset + ".txt"), "r") as f:
        for line in f:
            enrolments.append([int(x) for x in line.split()])

    with open(os.path.join(path, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\titem_id_list:token_seq\titem_length:float\n")
        for user, course in enrolments:
            tmp = user_timestamp.get(user, 0)
            hist = user_history.get(user, [])
            if(len(hist) == 0):
                user_history[user] = ['0']
            hist_str = ' '.join(user_history[user])
            f.write(f"{users[user]}\t{courses[course]}\t1\t{tmp}\t{hist_str}\t{len(user_history[user])}\n")
            user_timestamp[user] = tmp + 1
            user_history[user] = user_history.get(user, []) + [courses[course]]


def format_pgpr_mooc(datadir):
    """Format PGPR-MOOC dataset to recbole format.

    Args:
        datadir (str): path to the dataset
    """
    dataset_name = os.path.basename(os.path.normpath(datadir))
    users = save_users(datadir, dataset_name)
    courses = save_courses(datadir, dataset_name)

    subsets = ['train', "train_target", "test_target"]
    user_timestamp = {}
    user_history = {}
    for subset in subsets:
        save_enrolment(datadir, dataset_name, subset, users, courses, user_timestamp=user_timestamp, user_history=user_history)

    save_course_entity(datadir, dataset_name, courses)
    kg_triplets = []
    read_course_concept(datadir, kg_triplets, courses)
    read_course_teachers(datadir, kg_triplets, courses)
    read_course_schools(datadir, kg_triplets, courses)
    save_kg_triplets(kg_triplets, datadir, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    args = parser.parse_args()

    format_pgpr_mooc(args.datadir)
