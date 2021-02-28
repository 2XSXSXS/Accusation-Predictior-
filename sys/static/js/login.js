var $$ = mdui.JQ
var theWidth = $$(window).width();

// $$(document).ready(function () {
//     RelativeLogin();
// });


function RelativeLogin() {
    // var theWidth = $$(window).width();
    // alert(theWidth);
    // $$("#operate").width(theWidth)
    // alert($$("#operate").width())

}

function SetZero() {
    $$('#login-card').css('opacity', '0')
    $$('#register-card').css('opacity', '0')
    $$('#forget-card').css('opacity', '0')
}

// function showLogin() {
//     $("#login-card").fadeIn(200);
//     $("#register-card").fadeOut(0)
//     $("#forget-card").fadeOut(0)
// }
//
// function showRegister() {
//     $("#login-card").fadeOut(0)
//     $("#register-card").fadeIn(200);
//     $("#forget-card").fadeOut(0)
// }
//
// function showForget() {
//     $("#login-card").fadeOut(0)
//     $("#register-card").fadeOut(0)
//     $("#forget-card").fadeIn(200);
// }


function showLogin() {
    SetZero();
    $$("#operate1").width(theWidth)
    $$("#operate1").css('z-index', '0');
    $$("#operate2").css('z-index', '-1');
    $$("#operate3").css('z-index', '-1');
    $$("#login-card").css('opacity', '1');
}

function showRegister() {
    SetZero();
    $$("#operate2").width(theWidth);
    $$("#operate1").css('z-index', '-1');
    $$("#operate2").css('z-index', '0');
    $$("#operate3").css('z-index', '-1');
    $$("#register-card").css('opacity', '1');
}

function showForget() {
    SetZero();
    $$("#operate3").width(theWidth);
    $$("#operate1").css('z-index', '-1');
    $$("#operate2").css('z-index', '-1');
    $$("#operate3").css('z-index', '0');
    $$("#forget-card").css('opacity', '1');
}
