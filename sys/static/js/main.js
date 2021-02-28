var $$ = mdui.JQ;

$$(document).ready(function () {
   relative();
  });


$$(window).on('scroll', function() {
    scroll();
});

// var instLoginDialog = new mdui.Dialog('#login-dialog');
// var instRegisterDialog1 = new mdui.Dialog('#register-dialog-1') 
// var instRegisterDialog2 = new mdui.Dialog('#register-dialog-2')
// var instRegisterDialog3 = new mdui.Dialog('#register-dialog-3')
// // method
// document.getElementById('login-btn').addEventListener('click', function () {
//     instLoginDialog.open();
// });
// document.getElementById('relogin1-btn').addEventListener('click', function () {
//     instRegisterDialog1.close();
//     instLoginDialog.open();
// });
// document.getElementById('relogin2-btn').addEventListener('click', function () {
//     instRegisterDialog2.close();
//     instLoginDialog.open();
// });
// document.getElementById('relogin3-btn').addEventListener('click', function () {
//     instRegisterDialog3.close();
//     instLoginDialog.open();
// });
// document.getElementById('register-btn').addEventListener('click', function () {
//     instLoginDialog.close()
//     instRegisterDialog1.open();
// });
// document.getElementById('register1-btn').addEventListener('click', function () {
//     instRegisterDialog1.close()
//     instRegisterDialog2.open();
// });

// document.getElementById('register2-btn').addEventListener('click', function () {
//     instRegisterDialog2.close()
//     instRegisterDialog3.open();
// });
// document.getElementById('register-last2-btn').addEventListener('click', function () {
//     instRegisterDialog2.close()
//     instRegisterDialog1.open();
// });
// document.getElementById('register-last3-btn').addEventListener('click', function () {
//     instRegisterDialog3.close()
//     instRegisterDialog2.open();
// });

//设置relative高度
function relative() {
    var theWidth = $$(window).width();
    var theHeight = $$(window).height()/2;
    $('.div-relative').css({
        'width': theWidth,
        'height': theHeight
    });
}
//搜索框
function searchToggle(obj, evt) {
    var container = $(obj).closest('.search-wrapper');

    if (!container.hasClass('active')) {
        container.addClass('active');
        evt.preventDefault();
    }
    else if (container.hasClass('active') && $(obj).closest('.input-holder').length == 0) {
        container.removeClass('active');
        // clear input
        container.find('.search-input').val('');
        // clear and hide result container when we press close
        container.find('.result-container').fadeOut(100, function () { $(this).empty(); });
    }
}

// function submitFn(obj, evt) {
//     value = $(obj).find('.search-input').val().trim();
//
//     _html = "您搜索的关键词： ";
//     if (!value.length) {
//         _html = "关键词不能为空。";
//     }
//     else {
//         _html += "<b>" + value + "</b>";
//     }
//
//     $(obj).find('.result-container').html('<span>' + _html + '</span>');
//     $(obj).find('.result-container').fadeIn(100);
//
//     evt.preventDefault();
// }
// 状态栏滚动变色
function scroll() {
    var top = $$('#article').position().top + $$(window).height()/2-400;//获取导航栏变色的位置距顶部的高度
    var scrollTop = $(window).scrollTop();//获取当前窗口距顶部的高度
    if (scrollTop < top) {
        $$('#head-toolbar').addClass('mdui-shadow-0')
        if($$('body').hasClass('mdui-theme-layout-dark')) {
            $$('#head-toolbar').removeClass('mdui-shadow-3 mdui-color-black')
        } else {
            $$('#head-toolbar').removeClass('mdui-shadow-3 mdui-color-white')
        }
    } else {
        $$('#head-toolbar').removeClass('mdui-shadow-0')
        if($$('body').hasClass('mdui-theme-layout-dark')) {
            $$('#head-toolbar').addClass('mdui-shadow-3 mdui-color-black')
        } else {
            $$('#head-toolbar').addClass('mdui-shadow-3 mdui-color-white')
        }
    }
}
// 更改日间\夜间主题
document.getElementById('change-theme-btn').addEventListener('click', function () {
    if($$('body').hasClass('mdui-theme-layout-dark')) {
        $$('body').removeClass('mdui-theme-layout-dark')
        if($$('#head-toolbar').hasClass('mdui-color-black')) {
            $$('#head-toolbar').removeClass('mdui-color-black')
            $$('#head-toolbar').addClass('mdui-color-white')
        }
    }
    else {
        $$('body').addClass('mdui-theme-layout-dark')
        if($$('#head-toolbar').hasClass('mdui-color-white')) {
            $$('#head-toolbar').removeClass('mdui-color-white')
            $$('#head-toolbar').addClass('mdui-color-black')
        }
    }
});

function searchopen() {
    var btn = document.getElementById('search-btn');
    btn.click();
}
function logout() {
    var btn = document.getElementById('logout-btn');
    btn.click();
}