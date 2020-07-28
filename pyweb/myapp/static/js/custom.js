jQuery(function($) {
	$("#board_searchBtn").click(function(){
		var search_value = $("#board_search").val()
		if(!$.trim(search_value)){
			alert("검색어를 입력해주세요.")
		}
		else{
			location.href = "board?value="+search_value
		}
	})
	
	
	
	'use strict';
	
	$(".loader").delay(1000).fadeOut("slow");
  $("#overlayer").delay(1000).fadeOut("slow");	
	var siteMenuClone = function() {

		$('.js-clone-nav').each(function() {
			var $this = $(this);
			$this.clone().attr('class', 'site-nav-wrap').appendTo('.site-mobile-menu-body');
		});


		setTimeout(function() {
			
			var counter = 0;
      $('.site-mobile-menu .has-children').each(function(){
        var $this = $(this);
        
        $this.prepend('<span class="arrow-collapse collapsed">');

        $this.find('.arrow-collapse').attr({
          'data-toggle' : 'collapse',
          'data-target' : '#collapseItem' + counter,
        });

        $this.find('> ul').attr({
          'class' : 'collapse',
          'id' : 'collapseItem' + counter,
        });

        counter++;

      });

    }, 1000);

		$('body').on('click', '.arrow-collapse', function(e) {
      var $this = $(this);
      if ( $this.closest('li').find('.collapse').hasClass('show') ) {
        $this.removeClass('active');
      } else {
        $this.addClass('active');
      }
      e.preventDefault();  
      
    });

		$(window).resize(function() {
			var $this = $(this),
				w = $this.width();

			if ( w > 768 ) {
				if ( $('body').hasClass('offcanvas-menu') ) {
					$('body').removeClass('offcanvas-menu');
				}
			}
		})

		$('body').on('click', '.js-menu-toggle', function(e) {
			var $this = $(this);
			e.preventDefault();

			if ( $('body').hasClass('offcanvas-menu') ) {
				$('body').removeClass('offcanvas-menu');
				$this.removeClass('active');
			} else {
				$('body').addClass('offcanvas-menu');
				$this.addClass('active');
			}
		}) 

		// click outisde offcanvas
		$(document).mouseup(function(e) {
	    var container = $(".site-mobile-menu");
	    if (!container.is(e.target) && container.has(e.target).length === 0) {
	      if ( $('body').hasClass('offcanvas-menu') ) {
					$('body').removeClass('offcanvas-menu');
				}
	    }
		});
	}; 
	siteMenuClone();


	var sitePlusMinus = function() {
		$('.js-btn-minus').on('click', function(e){
			e.preventDefault();
			if ( $(this).closest('.input-group').find('.form-control').val() != 0  ) {
				$(this).closest('.input-group').find('.form-control').val(parseInt($(this).closest('.input-group').find('.form-control').val()) - 1);
			} else {
				$(this).closest('.input-group').find('.form-control').val(parseInt(0));
			}
		});
		$('.js-btn-plus').on('click', function(e){
			e.preventDefault();
			$(this).closest('.input-group').find('.form-control').val(parseInt($(this).closest('.input-group').find('.form-control').val()) + 1);
		});
	};
	// sitePlusMinus();

   var siteIstotope = function() {
  	/* activate jquery isotope */
	  var $container = $('#posts').isotope({
	    itemSelector : '.item',
	    isFitWidth: true
	  });

	  $(window).resize(function(){
	    $container.isotope({
	      columnWidth: '.col-sm-3'
	    });
	  });
	  
	  $container.isotope({ filter: '*' });

	    // filter items on button click
	  $('#filters').on( 'click', 'button', function(e) {
	  	e.preventDefault();
	    var filterValue = $(this).attr('data-filter');
	    $container.isotope({ filter: filterValue });
	    $('#filters button').removeClass('active');
	    $(this).addClass('active');
	  });
  }

  siteIstotope();

  var fancyBoxInit = function() {
	  $('.fancybox').on('click', function() {
		  var visibleLinks = $('.fancybox');

		  $.fancybox.open( visibleLinks, {}, visibleLinks.index( this ) );

		  return false;
		});
	}
	fancyBoxInit();


	var stickyFillInit = function() {
		$(window).on('resize orientationchange', function() {
	    recalc();
	  }).resize();

	  function recalc() {
	  	if ( $('.jm-sticky-top').length > 0 ) {
		    var elements = $('.jm-sticky-top');
		    Stickyfill.add(elements);
	    }
	  }
	}
	stickyFillInit();


	// navigation
  var OnePageNavigation = function() {
    var navToggler = $('.site-menu-toggle');
   	$("body").on("click", ".main-menu li a[href^='#'], .smoothscroll[href^='#'], .site-mobile-menu .site-nav-wrap li a", function(e) {
      e.preventDefault();

      var hash = this.hash;

      $('html, body').animate({
        'scrollTop': $(hash).offset().top
      }, 600, 'easeInOutCirc', function(){
        window.location.hash = hash;
      });

    });
  };
  OnePageNavigation();

  var counterInit = function() {
		if ( $('.section-counter').length > 0 ) {
			$('.section-counter').waypoint( function( direction ) {

				if( direction === 'down' && !$(this.element).hasClass('ftco-animated') ) {

					var comma_separator_number_step = $.animateNumber.numberStepFactories.separator(',')
					$('.number').each(function(){
						var $this = $(this),
							num = $this.data('number');
							console.log(num);
						$this.animateNumber(
						  {
						    number: num,
						    numberStep: comma_separator_number_step
						  }, 7000
						);
					});
					
				}

			} , { offset: '95%' } );
		}

	}
	counterInit();

	var selectPickerInit = function() {
		$('.selectpicker').selectpicker();
	}
	selectPickerInit();

	var owlCarouselFunction = function() {
		$('.single-carousel').owlCarousel({
	    loop:true,
	    margin:0,
	    nav:true,
	    autoplay: true,
	    items:1,
	    nav: false,
	    smartSpeed: 1000
		});

	}
	owlCarouselFunction();

	var quillInit = function() {

		var toolbarOptions = [
		  ['bold', 'italic', 'underline', 'strike'],        // toggled buttons
		  ['blockquote', 'code-block'],

		  [{ 'header': 1 }, { 'header': 2 }],               // custom button values
		  [{ 'list': 'ordered'}, { 'list': 'bullet' }],
		  [{ 'script': 'sub'}, { 'script': 'super' }],      // superscript/subscript
		  [{ 'indent': '-1'}, { 'indent': '+1' }],          // outdent/indent
		  [{ 'direction': 'rtl' }],                         // text direction

		  [{ 'size': ['small', false, 'large', 'huge'] }],  // custom dropdown
		  [{ 'header': [1, 2, 3, 4, 5, 6, false] }],

		  [{ 'color': [] }, { 'background': [] }],          // dropdown with defaults from theme
		  [{ 'font': [] }],
		  [{ 'align': [] }],

		  ['clean']                                         // remove formatting button
		];

		if ( $('.editor').length > 0 ) {
			var quill = new Quill('#editor-1', {
			  modules: {
			    toolbar: toolbarOptions,
			  },
			  placeholder: 'Compose an epic...',
			  theme: 'snow'  // or 'bubble'
			});
			var quill = new Quill('#editor-2', {
			  modules: {
			    toolbar: toolbarOptions,
			  },
			  placeholder: 'Compose an epic...',
			  theme: 'snow'  // or 'bubble'
			});
		}

	}
	quillInit();
  
});

function spacebarDisable() {
	var code = event.keyCode;
	if(code == 32) event.returnValue = false;
}


function board_write() {
	
  var name = $("#board_write_name").val()
  var passwd = $("#board_write_passwd").val()
  var title = $("#board_write_title").val()
  var text = $("#board_write_text").val()

  if (!name){
	  alert("닉네임을 입력하세요.")
  }
  else if (!passwd) {
	  alert('비밀번호를 입력하세요.')
  }
  else if (!$.trim(title)) {
	  alert('글 제목을 입력하세요.')
  }
  else {
	  $("#board_write_frm").submit()
  }
		  
	  
	
}

function insert_comment(){
	var name = $("#comment_name").val()
	var passwd = $("#comment_passwd").val()
	var commnet = $("#comment").val()
	
	if (!name){
		alert("닉네임을 입력하세요.")
	}
	else if (!passwd) {
		alert('비밀번호를 입력하세요.')
	}
	else if (!$.trim(comment)) {
		alert('댓글 내용을 입력하세요.')
	}
	else {
		$("#comment_frm").submit()
	}
	
	
}


function board_update_save() {
	
	 alert("수정되었습니다.")
	 $("#board_update_frm").submit()
	  
		
}

function board_update(board_no) {
	location.href = "board_update?board_no="+board_no	
}

function board_delete(board_no) {
	location.href = "board_delete?board_no="+board_no
	alert("삭제되었습니다.")
}

function board_passwd_check_dialog(type) {
	var type = type
	$("#board_passwd_check_dialog").dialog({
		title : "Check",
		width : 330,
		height : 250,
		resizable : false,
		draggable : false,
		modal : true,
		buttons : {
			확인 : function(){
				board_passwd_check(type)
			},
			
			닫기 : function() {
				$("#board_passwd_check_dialog").dialog("close");
			}
		}
	});
	
}
function comment_passwd_check_dialog(comment_no) {
	$("#comment_passwd_check_dialog").dialog({
		title : "Check",
		width : 330,
		height : 250,
		resizable : false,
		draggable : false,
		modal : true,
		buttons : {
			확인 : function(){
				comment_passwd_check(comment_no)
			},
			
			닫기 : function() {
				$("#comment_passwd_check_dialog").dialog("close");
			}
		}
	});
	
}

function comment_passwd_check(comment_no) { 
	var comment_passwd2 = $("#comment_passwd2").val();
	var type = type
	$.ajax({
		type : "post",
		url : "comment_passwd_check",
		data : {
			"comment_no" : comment_no,
			"comment_passwd2" : comment_passwd2
		},
		dataType : "json",
		success : function(data) {
			if (data.result == "success") {
				$("#comment_passwd_check_dialog").dialog("close");
				location.href="comment_delete?comment_no="+comment_no+"&board_no="+$("#board_no").val()
			} else {
				$("#comment_passwd_check_div").html("비밀번호를 다시 입력하세요.");
			}
		},
		error : function(e) {
			alert("에러" + e);
		}
	});
}


function board_passwd_check(type) { 
	var board_no = $("#board_no").val();
	var board_passwd = $("#board_passwd").val();
	var type = type
	$.ajax({
		type : "post",
		url : "board_passwd_check",
		data : {
			"board_no" : board_no,
			"board_passwd" : board_passwd
		},
		dataType : "json",
		success : function(data) {
			if (data.result == "success") {
				if(type == 1){
					$("#board_passwd_check_dialog").dialog("close");
					board_update(board_no)
				}
				else if (type == 2){
					$("#board_passwd_check_dialog").dialog("close");
					board_delete(board_no)
				}
				
			} else {
				$("#board_passwd_check_div").html("비밀번호를 다시 입력하세요.");
			}
		},
		error : function(e) {
			alert("에러" + e);
		}
	});
}

//ajax post csrf 설정
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});