import os
import argparse


def save_learners(path, name):
    """Save coco learners to .user file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """
    users = []
    with open(os.path.join(path, "learners.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(path, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"{user_id}\n")
    return users


def save_courses(path, name):
    """Save coco courses to .item file.

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
        for item_id in courses:
            f.write(f"{item_id}\n")
    return courses


def save_course_entity(path, name, courses):
    """Save coco course entity to .link file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        courses (list): list of courses
    """
    with open(os.path.join(path, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for course in courses:
            f.write(f"{course}\tE_{course}\n")


def read_course_instructors(path, kg_triplets, courses):
    """Update kg triplets with instructors.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_instructor.txt"), "r") as f:
        for i, line in enumerate(f):
            instructor = line.strip()
            if instructor:
                kg_triplets.append(
                    [
                        "E_" + courses[int(i)],
                        "instructor",
                        "Instructor_" + instructor,
                    ]
                )


def read_course_category(path, kg_triplets, courses):
    """Update kg triplets to with category.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_scategory.txt"), "r") as f:
        for i, line in enumerate(f):
            category = line.strip()
            if category:
                kg_triplets.append(
                    [
                        "E_" + courses[int(i)],
                        "category",
                        "Category_" + category,
                    ]
                )


def read_course_skills(path, kg_triplets, courses):
    """Update kg triplets with skills.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_skills.txt"), "r") as f:
        for i, line in enumerate(f):
            skills = line.strip()
            if skills:
                for skill in skills.split():
                    kg_triplets.append(
                        [
                            "E_" + courses[int(i)],
                            "skill",
                            "Skill_" + skill,
                        ]
                    )


def read_category_hierarchy(path, kg_triplets):
    """Update kg triplets to with category hierarchy.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "scategory_fcategory.txt"), "r") as f:
        for i, line in enumerate(f):
            pcategory = line.strip()
            if pcategory:
                kg_triplets.append(
                    [
                        "Category_" + str(i),
                        "child_category",
                        "ParentCategory_" + pcategory,
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


def format_pgpr_coco(datadir):
    """Format PGPR-COCO dataset to recbole format.

    Args:
        datadir (str): path to the dataset
    """
    dataset_name = os.path.basename(os.path.normpath(datadir))
    users = save_learners(datadir, dataset_name)
    courses = save_courses(datadir, dataset_name)

    subsets = ['train', "train_target", "test_target"]
    user_timestamp = {}
    user_history = {}
    for subset in subsets:
        save_enrolment(datadir, dataset_name, subset, users, courses, user_timestamp=user_timestamp, user_history=user_history)

    save_course_entity(datadir, dataset_name, courses)
    kg_triplets = []
    read_course_instructors(datadir, kg_triplets, courses)
    read_course_category(datadir, kg_triplets, courses)
    read_course_skills(datadir, kg_triplets, courses)
    # read_category_hierarchy(datadir, kg_triplets)
    save_kg_triplets(kg_triplets, datadir, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    args = parser.parse_args()

    format_pgpr_coco(args.datadir)
