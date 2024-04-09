const express = require("express");
const fs = require("fs");
const path = require("path");
const url = require("url");

const app = express();

const port = 8080;
const public = "public" // Folder for public files

function serveFile(res, pathName, mime) {
    mime = mime || "text/html";
    fs.readFile(path.join(__dirname, pathName), (err,data) => {
        if (err) {
            res.writeHead(500, {"Content-Type": "text/plain"});
            return res.end(`Error loading ${pathName} with Error: ${err}`);
        }

        res.writeHead(200, {"Content-Type": mime});
        res.end(data);
    })
}

app.use(express.static(path.join(__dirname, public)));

app.get("/img/chesspieces/wikipedia/*.png", async(req, res) => {
    res.redirect("https://chessboardjs.com/" + req.url);
});

app.get("/*.*", async(req, res) => {
    var options = url.parse(req.url, true);
    serveFile(res, options.pathname, "text/plain");
});

app.listen(port, () => {
    console.log(`Server successfully running on port ${port}`)
})