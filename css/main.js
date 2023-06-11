document.getElementById("main").parentNode.childNodes[0].classList.add("header_bar");
document.getElementById("main").parentNode.style = "padding: 0; margin: 0";
document.getElementById("main").parentNode.parentNode.parentNode.style = "padding: 0";


let main = document.getElementById('main');
let main_parent = main.parentNode;
let extensions = document.getElementById('extensions');


main_parent.addEventListener('click', function(e) {
    
    if (main.offsetHeight > 0 && main.offsetWidth > 0) {
        extensions.style.display = 'flex';
    } else {
        extensions.style.display = 'none';
    }
});
