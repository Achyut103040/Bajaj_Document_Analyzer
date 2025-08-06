# Heroku Deployment Steps

## After Installing Heroku CLI:

1. **Login to Heroku:**
   ```
   heroku login
   ```

2. **Create a new Heroku app:**
   ```
   heroku create bajaj-document-analyzer
   ```
   (Or use any unique name you prefer)

3. **Verify remote was added:**
   ```
   git remote -v
   ```
   You should see both origin (GitHub) and heroku remotes

4. **Commit any changes:**
   ```
   git add .
   git commit -m "Ready for Heroku deployment"
   ```

5. **Deploy to Heroku:**
   ```
   git push heroku main
   ```

6. **Scale the web dyno:**
   ```
   heroku ps:scale web=1
   ```

7. **Open your app:**
   ```
   heroku open
   ```

8. **Check logs if issues:**
   ```
   heroku logs --tail
   ```

## Your app will be available at:
- https://your-app-name.herokuapp.com/
- Test endpoint: https://your-app-name.herokuapp.com/hackrx/run

## Environment Variables:
If you need to set environment variables on Heroku:
```
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set PINECONE_API_KEY=your_key_here
heroku config:set DEVICE=cpu
```
