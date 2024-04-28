var canvas  = $("#canvas"),
    context = canvas.get(0).getContext("2d"),
    $result = $('#result');

var croppedImageDataURL
$('#fileInput').on( 'change', function(){
    if (this.files && this.files[0]) {
      if ( this.files[0].type.match(/^image\//) ) {
        var reader = new FileReader();
        reader.onload = function(evt) {
           var img = new Image();
           img.onload = function() {
             context.canvas.height = img.height;
             context.canvas.width  = img.width;
             context.drawImage(img, 0, 0);
             var cropper = canvas.cropper({
//               aspectRatio: 16 / 9
//               aspectRatio: 9 / 9

             });
             canvas.cropper('reset');
             $result.empty();
             $("[id*='Parasite']").text("");


             $('#btnCrop').click(function() {
                // Get a string base 64 data url
                croppedImageDataURL = canvas.cropper('getCroppedCanvas').toDataURL("image/png");
//                console.log("printing croppedImageDataURL\n");

//                console.log(croppedImageDataURL);
//                console.log("\n\n");

                var croppedImageDataURL_0 = canvas.cropper('getCroppedCanvas');
//                console.log(croppedImageDataURL);

                $result.empty();
                $result.append( $('<img>').attr('src', croppedImageDataURL) );
                $("[id*='Parasite']").text("");
//                var result_canvas  = $("#result");
//                res_context = result_canvas.get(0).getContext("2d");
//                res_context.drawImage(croppedImageDataURL, 0, 0);

             });
             $('#btnRestore').click(function() {
               canvas.cropper('reset');
               $result.empty();
               $("[id*='Parasite']").text("");

             });
           };
           img.src = evt.target.result;
				};
        reader.readAsDataURL(this.files[0]);
      }
      else {
        alert("Invalid file type! Please select an image file.");
      }
    }
    else {
      alert('No file(s) selected.');
    }
});


$(function(){

    $("[id*='validateParasite']").click(function(){


//	$('button').click(function(){
//		var user = $('#txtUsername').val();
//		var pass = $('#txtPassword').val();
		$.ajax({
			url: '/parasiteDetection',
//			data: $('form').serialize(),
//			data:croppedImageDataURL,
			data: { information : "Croppedimage" , userdata : croppedImageDataURL },
            dataType: "json",
			type: 'POST',
			success: function(response){
//				console.log(response);
                var status=response["status"]
				$("[id*='Parasite']").text(response["parasite"]);
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});



//function toServer(usrdata){
//    $.ajax({
//        url: "/parasiteDetection",
//        type: "POST",
//        data: { information : "You have a very nice website, sir." , userdata : usrdata },
//        dataType: "json",
////        success: function(data) {
////           //s <!-- do something here -->
////            $('#somediv').html(data);
////        }
//        success: function(data){
//				console.log(data);
//		},
//
//        error: function(error){
//				console.log(error);
//		}
//
//        });
//    }


