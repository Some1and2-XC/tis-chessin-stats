var board = null;
var game = new Chess();
var $status = $("#status");
var $eco = $("#eco");
var $opening = $("#opening");
var $fen = $("#fen");
var $pgn = $("#pgn");
var $eval = $("#eval");

var $white_elo = $("#whiteElo");
var $black_elo = $("#blackElo");

var $white_res = $("#res-1-0");
var $black_res = $("#res-0-1");
var $draw_res = $("#res-draw");

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;

    if ((game.turn() === "w" && piece.search(/^b/) !== -1) ||
        (game.turn() === "b" && piece.search(/^w/) !== -1)) {
        return false
    }
}

function onDrop(source, target) {
    var move = game.move({
        from: source,
        to: target,
        promotion: "q",
    });

    if (move === null) return "snapback";

    updateStatus();

    return move;
}

function onSnapEnd() {
    board.position(game.fen());
}

function updateStatus() {
    var status = "";

    var moveColor = "White";

    if (game.turn() === "b") {
        moveColor = "Black";
    }

    if (game.in_checkmate()) {
        status = `Game over, ${moveColor} is in checkmate. `;
    }

    else if (game.in_draw()) {
        status = "Game over, drawn position";
    }

    else {
        status = `${moveColor} to move`;

        if (game.in_check()) {
            status += `, ${moveColor} is in check. `
        }
    }

    $status.html(status);
    $fen.html(game.fen());
    $pgn.html(game.pgn());

    fetch("./get_stats?" + new URLSearchParams({
        whiteElo: $white_elo.val(),
        blackElo: $black_elo.val(),
        pgn: game.pgn(),
    }))
        .then(res => res.json())
        .then(data => {
            $opening.html(data.opening)
            $eco.html(data.eco)
            $eval.html(data.eval / 100);

            const HEIGHT_FACTOR = 1;
            const PROPERTY = "flex-grow"

            $white_res.css(PROPERTY, data.prediction["1-0"] * HEIGHT_FACTOR);
            $white_res.html(`<p>${(data.prediction["1-0"] * 100).toPrecision(2)}%</p>`);

            $black_res.css(PROPERTY, data.prediction["0-1"] * HEIGHT_FACTOR);
            $black_res.html(`<p>${(data.prediction["0-1"] * 100).toPrecision(2)}%</p>`);
            $black_res.css("color", "white");

            $draw_res.css(PROPERTY, data.prediction["1/2-1/2"] * HEIGHT_FACTOR);
            $draw_res.html(`<p>${(data.prediction["1/2-1/2"] * 100).toPrecision(2)}%</p>`);

            return data;
        })
        .then(console.log)
        ;

}

var config = {
    draggable: true,
    position: "start",
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
}

board = ChessBoard("board1", config)

updateStatus()
