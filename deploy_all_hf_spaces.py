#!/usr/bin/env python3
"""
Deploy all updated HF spaces to Hugging Face
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result"""
    try:
        print(f"🔧 Running: {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        
        if result.stdout:
            print(f"📝 Output: {result.stdout.strip()}")
        
        if result.returncode != 0 and check:
            print(f"❌ Error: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_git_config():
    """Check if git is configured"""
    print("🔍 Checking git configuration...")
    
    # Check if git is installed
    if not run_command("git --version", check=False):
        print("❌ Git is not installed or not in PATH")
        return False
    
    # Check git config
    result = subprocess.run("git config --global user.name", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("⚠️  Git user.name not configured")
        name = input("Enter your git username: ")
        run_command(f'git config --global user.name "{name}"')
    
    result = subprocess.run("git config --global user.email", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("⚠️  Git user.email not configured")
        email = input("Enter your git email: ")
        run_command(f'git config --global user.email "{email}"')
    
    return True

def deploy_space(space_name, repo_url=None):
    """Deploy a single HF space"""
    print(f"\n🚀 Deploying {space_name}...")
    print("=" * 50)
    
    space_path = Path(space_name)
    if not space_path.exists():
        print(f"❌ Space directory {space_name} not found")
        return False
    
    # Change to space directory
    original_cwd = os.getcwd()
    os.chdir(space_path)
    
    try:
        # Initialize git if not already initialized
        if not Path(".git").exists():
            print("📁 Initializing git repository...")
            if not run_command("git init"):
                return False
        
        # Set up remote repository URL
        repo_url = repo_url or f"https://huggingface.co/spaces/Syedkaif29/PdfTransic"
        
        # Check if remote origin exists
        remote_exists = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
        
        if "origin" not in remote_exists.stdout:
            print(f"🔗 Adding remote origin: {repo_url}")
            if not run_command(f"git remote add origin {repo_url}"):
                return False
        else:
            # Update remote URL if it exists but might be incorrect
            print(f"🔄 Updating remote origin URL: {repo_url}")
            run_command(f"git remote set-url origin {repo_url}", check=False)
        
        # Add all files
        print("📄 Adding files to git...")
        if not run_command("git add ."):
            return False
        
        # Check if there are changes to commit
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("ℹ️  No changes to commit")
            return True
        
        # Commit changes
        commit_msg = f"Deploy {space_name} with latest improvements - Memory optimization, Enhanced PDF processing, OCR support"
        print("💾 Committing changes...")
        if not run_command(f'git commit -m "{commit_msg}"'):
            return False
        
        # Detect current branch
        branch_result = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
        current_branch = branch_result.stdout.strip()
        
        if not current_branch:
            # If no branch detected, check if we're in detached HEAD state
            # Default to 'main' if we can't determine the branch
            print("⚠️ Could not detect current branch, defaulting to 'main'")
            current_branch = "main"
            # Create and checkout main branch
            run_command("git checkout -b main", check=False)
        
        print(f"🔍 Current branch: {current_branch}")
        
        # Pull latest changes first to avoid conflicts
        print("🔄 Pulling latest changes from Hugging Face...")
        
        # Try to stash any local changes first to avoid conflicts
        run_command("git stash", check=False)
        
        # Try pulling with strategy option to handle conflicts automatically
        pull_success = run_command(f"git pull origin {current_branch} --strategy-option=theirs", check=False)
        
        # If pull fails, try the alternative branch
        if not pull_success:
            alternative_branch = "master" if current_branch == "main" else "main"
            print(f"🔄 Trying to pull from {alternative_branch} branch...")
            alt_pull_success = run_command(f"git pull origin {alternative_branch} --strategy-option=theirs", check=False)
            
            # If alternative pull succeeds, switch to that branch
            if alt_pull_success:
                print(f"🔀 Switching to {alternative_branch} branch")
                run_command(f"git checkout -b {alternative_branch}", check=False)
                current_branch = alternative_branch
                
        # Try to apply stashed changes back
        run_command("git stash pop", check=False)
        
        # Push to HF using the detected/selected branch
        print(f"📤 Pushing to Hugging Face ({current_branch} branch)...")
        if not run_command(f"git push origin {current_branch}"):
            # Try pushing to alternative branch if current branch fails
            alternative_branch = "master" if current_branch == "main" else "main"
            print(f"🔄 Trying to push to {alternative_branch} branch...")
            if not run_command(f"git push origin {alternative_branch}"):
                # Try force pushing as last resort
                print("⚠️ Normal push failed, trying force push (use with caution)...")
                if not run_command(f"git push -f origin {current_branch}", check=False):
                    return False
        
        print(f"✅ Successfully deployed {space_name}!")
        return True
        
    except Exception as e:
        print(f"❌ Error deploying {space_name}: {e}")
        return False
    
    finally:
        os.chdir(original_cwd)

def main():
    """Main deployment function"""
    print("🌍 Hugging Face Spaces Deployment Tool")
    print("=" * 60)
    
    # Check git configuration
    if not check_git_config():
        print("❌ Git configuration failed")
        return False
    
    # Define spaces to deploy
    spaces = [
        {
            "name": "hf_space_deploy",
            "description": "Main deployment space"
        },
        {
            "name": "hf_space_fix", 
            "description": "Bug fixes and improvements"
        },
        {
            "name": "hf_space_memory_fix",
            "description": "Memory optimization version"
        },
        {
            "name": "hf_space_requirements_fix",
            "description": "Requirements and dependencies fix"
        }
    ]
    
    print(f"\n📋 Found {len(spaces)} spaces to deploy:")
    for i, space in enumerate(spaces, 1):
        print(f"  {i}. {space['name']} - {space['description']}")
    
    # Ask user which spaces to deploy
    print("\n🤔 Which spaces would you like to deploy?")
    print("1. Deploy all spaces")
    print("2. Select specific spaces")
    print("3. Deploy one by one (with confirmation)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    deployed_count = 0
    failed_count = 0
    
    if choice == "1":
        # Deploy all spaces
        print("\n🚀 Deploying all spaces...")
        for space in spaces:
            if deploy_space(space["name"]):
                deployed_count += 1
            else:
                failed_count += 1
    
    elif choice == "2":
        # Select specific spaces
        print("\n📝 Select spaces to deploy (comma-separated numbers):")
        selection = input("Enter space numbers: ").strip()
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            for idx in indices:
                if 0 <= idx < len(spaces):
                    if deploy_space(spaces[idx]["name"]):
                        deployed_count += 1
                    else:
                        failed_count += 1
                else:
                    print(f"❌ Invalid space number: {idx + 1}")
        except ValueError:
            print("❌ Invalid input format")
            return False
    
    elif choice == "3":
        # Deploy one by one with confirmation
        for space in spaces:
            confirm = input(f"\n🤔 Deploy {space['name']}? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                if deploy_space(space["name"]):
                    deployed_count += 1
                else:
                    failed_count += 1
            else:
                print(f"⏭️  Skipped {space['name']}")
    
    else:
        print("❌ Invalid choice")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully deployed: {deployed_count} spaces")
    print(f"❌ Failed deployments: {failed_count} spaces")
    
    if deployed_count > 0:
        print("\n🌐 Your spaces should be available at:")
        print("   https://huggingface.co/spaces/Syedkaif29/PdfTransic")
        print("\n⏳ Note: It may take a few minutes for spaces to build and start")
        
        print("\n🔧 Next steps:")
        print("1. Check your Hugging Face Spaces dashboard")
        print("2. Monitor build logs for any issues")
        print("3. Test the deployed APIs")
        print("4. Update space settings if needed")
    
    return deployed_count > 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)