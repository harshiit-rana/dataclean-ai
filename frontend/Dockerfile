FROM node:20-alpine

WORKDIR /app

# The lockfile too, and `npm ci`: the versions the tests passed against, installed as locked. With only package.json,
# `npm install` resolved the whole tree afresh -- not what was tested -- and on a clean build hit an npm resolver bug
# ("Cannot read properties of null (reading 'edgesOut')"), which the compose job in CI caught (#117).
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
