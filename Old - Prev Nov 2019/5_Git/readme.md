
# Intro to git

Week 1 | Lesson 4

### LEARNING OBJECTIVES

*After this lesson, you will be able to:*
- Use/explain git commands like init, add, commit, push, pull, and clone
- Distinguish between local and remote repositories
- Create, copy, and delete repositories locally, or on Github
- Clone remote repositories

### STUDENT PRE-WORK

*Before this lesson, you should already be able to:*
- Have completed [Code Academy: Learn Git](https://www.codecademy.com/learn/learn-git)
- Install [Homebrew](http://brew.sh/)
- Install git (after installing Homebrew, type "brew install git")
- Setup a GitHub account
- Setup [SSH key](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)

## Git vs GitHub and version control - Intro (20 mins)

First things first - Git is **not** Github. This is a common mistake that people make!

#### What is Git?

[Git](https://git-scm.com/) is:
- A program you run from the command line
- A distributed version control system
Programmers use Git so that they can keep the history of all the changes to their code. This means that they can rollback changes (or switch to older versions) as far back int time as they started using Git on their project.
A codebase in Git is referred to as a **repository**, or **repo**, for short.
Git was created by [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds), the principal developer of Linux.

#### What is Github?

[Github](https://github.com/) is:
- A hosting service for Git repositories
- A web interface to explore Git repositories
- A social network of programmers
- We all have individual accounts and put our codebases on our Github account
- You can follow users and star your favorite projects
- Developers can access codebases on other public accounts
- GitHub *uses* Git

#### Can you use git without Github?

Think about this quote: “Git is software. GitHub is a company that happens to use Git software.”  So yes, you can certainly use Git without GitHub!
Your local repository consists of three "trees" maintained by Git.
- **Working Directory**: which holds the actual files.
- **Index**: which acts as a staging area
- **HEAD**: which points to the last commit you've made.
![workflow](https://cloud.githubusercontent.com/assets/40461/8221736/f1f7e972-1559-11e5-9dcb-66b44139ee6f.png)

#### So many commands?!

There are a lot of commands you can use in git. You can take a look at a list of the available commands by running:
```bash
$ git help -a
```
Even though there are lots of commands, on the course we will really only need about 10.

## Let's use Git - Demo (15 mins)

First, create and navigate into a new directory on your Home directory:
```bash
$ cd ~
$ mkdir BV
$ cd BV
```
You can place this directory under Git revision control using the command:
```bash
$ git init
```
Git will reply:
```bash
Initialized empty Git repository in <location>
```
You've now initialized the working directory.

#### The .git folder

If we look at the contents of this empty folder using:
```bash
ls -A
```
> Check: What do you see?
We should see that there is now a hidden folder called `.git` this is where all of the information about your repository is stored. There is no need for you to make any changes to this folder. You can control all the git flow using `git` commands.

#### Add a file

Let's create a new file:
```bash
$ touch a.txt
```
If we run `git status` we should get:
```bash
On branch master
Initial commit
Untracked files:
  (use "git add <file>..." to include in what will be committed)
    a.txt
nothing added to commit but untracked files present (use "git add" to track)
```
This means that there is a new **untracked** file. Next, tell Git to take a snapshot of the contents of all files under the current directory (note the .)
```bash
$ git add .
```
After this, `git status` returns:
```bash
On branch master
Initial commit
Changes to be committed:
(use "git rm --cached <file>..." to unstage)
    new file:   a.txt
```
This snapshot is now stored in a temporary staging area which Git calls the "index".

#### Commit

To permanently store the contents of the index in the repository, (commit these changes to the HEAD), you need to run:
```bash
$ git commit -m "Add a.txt"
```
`Commit` is the command; the `-m` flag lets you add a "commit message" documenting your change, which you should always do. (If you don't use the `-m` flag, git will open a text editor and prompt you for a message.)
You should now get:
```bash
[master (root-commit) b4faebd] Add a.txt
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 a.txt
```

#### Checking the log

If we want to view the commit history, we can run:
```bash
git log
```
You should see something similar to:
```bash
commit ec1b1ebad5927e74d498ee588190686c51cab446
Author: <author>
Date:   <date>
Add a.txt
```

#### Making and cloning repositories - Demo (10 mins)

Let's do this together:
1. Go to your Github account, and click through to the "Repositories" tab on your profile
2. In the right hand side, hit the green + button for `New repository`
3. Name your repository `BV-projects`
4.  Check "Initialize this repository with a README"
5. Click the big green Create Repository button
We now need to connect our local Git repo with our remote repository on GitHub. We have to add a "remote" repository, an address where we can send our local files to be stored. Type the following in to the command line, replacing 'github-name' with your account name:

```bash
git remote add origin git@github.com:github-name/BV-projects.git
```

You've now initialized a local repo in which to share your projects, and connected it with a remote repo on GitHub which you own.
> Check: can you draw a diagram with the repos we're using so far? Add to this diagram the rest of the lesson.

#### Pushing to Github

In order to send files from our local machine to our remote repository on Github, we use the command `git push`. You also need to specify the name of the remote repo -- we called it `origin` -- and the name of the branch, in this case `master`.

```bash
git push origin master
```
> Check: What do you see?
This should fail due to new files on the remote repo. This is a common part of a collaborative project workflow: someone makes changes to the remote while you are working on the local version, so you need to bring **your version** up to date by *fetching* and *merging* the additions. After that you can push your own changes.

#### Pulling from Github

As we added the README.md in our remote repo, we need to first `pull` that file to our local repository to check that we haven't got a 'conflict'.

```bash
git pull origin master
```
```bash
git status
```
```bash
git add .
```
```bash
git commit -m "README.md"
```

Once we have done this, you should see the README file on your computer.

```bash
ls
```
Open it up and type some kind of modification/addition, then stage and commit it again. (This time we're committing the specific 
file rather than everything in the directory; in this case either works.)

```bash
git add README.md
git commit -m "Edit README.md"
```

Now you can push your changes:

```bash
git push origin master
```

Refresh your GitHub webpage, and the files should be there!

####  .gitignore

Currently we would upload every file in our git folder, but we may have many files which we do not wish to upload if they are specific to our local system and not relevant for others to see. This is particularly true if you want to swap between different operating systems, or if you have anything sensitive such as passwords stored locally that you do not want to push up to a public or semi public folder.
In order to control for this, we can create a .gitignore file and store any extension types that we wish to be ignored each on a separate row. A nice tool to do this is [gitignore.io](https://www.gitignore.io/)

####  Forking and cloning

Now that everyone has their first repository on GitHub, let's fork and clone a repository.
Forking gives you a copy of an existing repository -- you can make any changes you want without affecting the original. (You can also propose that the maintainer of the original merge your changes!).
Cloning gives you a local copy of a remote repository.
Let's fork and then clone our DSI course repo, which includes all the specifications for your projects.

#### Fork the repo!

While logged into GitHub, go to <insert your course URL here> and click the icon that says 'fork'. That's it!
To retrieve the contents of the repo, all you need to do is navigate back to Home and `clone`:

```bash
cd ~
$ git clone git@github.com:github-name/<your-course>.git
```

(You can also copy that git@... string from GitHub by clicking the green button saying "Clone or download".)
Git should reply:

```bash
Cloning into '<your-course>'...
remote: Counting objects: 3, done.
remote: Total 3 (delta 0), reused 3 (delta 0), pack-reused 0
Receiving objects: 100% (3/3), done.
Checking connectivity... done.
```

You now have cloned your first repository!

#### Syncing the repo

We also need a way to keep your new copy of <our course> up-to-date. Let's tell Git what the "upstream" repo is:
```bash
git remote add upstream git@github.com:belgravevalley/<your-course>.git
```
(Look closely: where exactly is the "upstream" repo?)
Now, you can update your local repo each morning with:

```bash
git pull upstream master
```

## Create a pull request on GitHub - Demo (5 mins)

Before you can open a pull request, you must create a branch in your local repository, commit to it, and push the branch to a repository or fork on GitHub.

1. Visit the repository you pushed to
2. Click the "Compare, review, create a pull request" button in the repository ![pr](https://cloud.githubusercontent.com/assets/40461/8229344/d344aa8e-15ad-11e5-8578-08893bcee335.jpg)
3. You'll land right onto the compare page - you can click Edit at the top to pick a new branch to merge in, using the Head Branch dropdown.
4. Select the target branch your branch should be merged to, using the Base Branch dropdown
5. Review your proposed change
6. Click "Click to create a pull request" for this comparison
7. Enter a title and description for your pull request
8. Click 'Send pull request'

## Assess - Independent Practice (10 mins)

- Show a partner how to use to:  init, add, commit, push, pull, and clone

## Conclusion (5 mins)

Feel comfortable with Git and GitHub? Since we'll be using it a lot of coursework, let's get
comfortable!

## Bonus Challenges

_These challenges are optional!_
Once you've mastered the basics, try furthering your understanding of Git by:
- Diving into the book ["Pro Git"](https://git-scm.com/book/en/v2).
- Considering best practices for Git commit messages by reading [this post](http://chris.beams.io/posts/git-commit/).
