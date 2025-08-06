#!/usr/bin/env python3
"""
🔍 HEROKU DEPLOYMENT ISSUE DIAGNOSIS
================================================================================
Diagnoses common Heroku deployment issues
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and return result"""
    print(f"\n🔍 {description}")
    print(f"   Command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return True, result.stdout
        else:
            print(f"   ❌ FAILED: {result.stderr.strip()}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False, str(e)

def check_heroku_issues():
    print("🚨 HEROKU DEPLOYMENT ISSUE DIAGNOSIS")
    print("=" * 60)
    
    issues_found = []
    
    # 1. Check if Heroku CLI is installed
    success, output = run_command("heroku --version", "Checking Heroku CLI")
    if not success:
        issues_found.append("Heroku CLI not installed")
    
    # 2. Check if logged into Heroku
    success, output = run_command("heroku auth:whoami", "Checking Heroku authentication")
    if not success:
        issues_found.append("Not logged into Heroku")
    
    # 3. Check if git is initialized
    success, output = run_command("git status", "Checking git repository")
    if not success:
        issues_found.append("Git repository not initialized")
    
    # 4. Check if Heroku remote exists
    success, output = run_command("git remote -v", "Checking Heroku remote")
    if success and "heroku" not in output:
        issues_found.append("Heroku remote not configured")
    
    # 5. Check essential files
    essential_files = ["Procfile", "requirements.txt", "runtime.txt", "api.py"]
    for file in essential_files:
        if not os.path.exists(file):
            issues_found.append(f"Missing {file}")
        else:
            print(f"   ✅ {file} exists")
    
    # 6. Check Procfile content
    if os.path.exists("Procfile"):
        with open("Procfile", "r") as f:
            content = f.read().strip()
            print(f"\n📄 Procfile content: {content}")
            if "web:" not in content:
                issues_found.append("Procfile missing web process")
            if "uvicorn" not in content and "gunicorn" not in content:
                issues_found.append("Procfile missing WSGI server")
    
    # 7. Check Python version
    if os.path.exists("runtime.txt"):
        with open("runtime.txt", "r") as f:
            python_version = f.read().strip()
            print(f"\n🐍 Python version: {python_version}")
            if not python_version.startswith("python-3."):
                issues_found.append("Invalid Python version in runtime.txt")
    
    # 8. Check if app was deployed
    success, output = run_command("heroku ps", "Checking dyno status")
    if success:
        if "web" not in output:
            issues_found.append("Web dyno not running")
    
    # 9. Check recent logs
    print(f"\n📋 Recent Heroku logs:")
    success, output = run_command("heroku logs --tail --num=20", "Getting recent logs")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("🔍 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if not issues_found:
        print("✅ No obvious issues found")
        print("\n🔧 NEXT STEPS:")
        print("1. Check if app was deployed: heroku ps")
        print("2. Check logs: heroku logs --tail")
        print("3. Restart dyno: heroku ps:restart")
        print("4. Open app: heroku open")
    else:
        print("❌ Issues found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n🛠️ RECOMMENDED FIXES:")
        print("-" * 30)
        
        if "Heroku CLI not installed" in issues_found:
            print("• Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli")
        
        if "Not logged into Heroku" in issues_found:
            print("• Login: heroku login")
        
        if "Git repository not initialized" in issues_found:
            print("• Initialize git: git init")
            print("• Add files: git add .")
            print("• Commit: git commit -m 'Initial commit'")
        
        if "Heroku remote not configured" in issues_found:
            print("• Create app: heroku create your-app-name")
            print("• Or add remote: heroku git:remote -a existing-app-name")
        
        if "Web dyno not running" in issues_found:
            print("• Deploy: git push heroku main")
            print("• Scale: heroku ps:scale web=1")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    success = check_heroku_issues()
    
    if success:
        print(f"\n🎉 READY FOR HEROKU DEPLOYMENT!")
    else:
        print(f"\n⚠️  PLEASE FIX ISSUES BEFORE DEPLOYMENT")
    
    sys.exit(0 if success else 1)
