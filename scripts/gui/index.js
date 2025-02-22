window.onload = function () {

    var rangeSlider = function () {
        var slider = $('.range-slider'),
            range = $('.range-slider__range'),
            value = $('.range-slider__value');

        slider.each(function () {

            value.each(function () {
                var value = $(this).prev().attr('value');
                $(this).html(value);
            });

            range.on('input', function () {
                var id = this.getAttribute('data-id');
                if (id) {
                    document.getElementById(id).setAttribute('data-style', this.value);
                } else {
                    this.setAttribute('image-id', this.value);
                }
                $(this).next(value).html(this.value);
                render_button();
            });
        });
    };

    rangeSlider();

    function dragMoveListener(event) {
        console.log('dragMobve');
        var target = event.target,
            // keep the dragged position in the data-x/data-y attributes
            x = (parseFloat(target.getAttribute('data-x')) || 0) + event.dx,
            y = (parseFloat(target.getAttribute('data-y')) || 0) + event.dy;

        // translate the element
        target.style.webkitTransform =
            target.style.transform =
                'translate(' + x + 'px, ' + y + 'px)';

        // update the posiion attributes
        target.setAttribute('data-x', x);
        target.setAttribute('data-y', y);
        selectItem(event, event.target);
        // render_button();
    }

// this is used later in the resizing and gesture demos
    window.dragMoveListener = dragMoveListener;

    interact('.resize-drag')
        .draggable({
            onmove: window.dragMoveListener,
            onend: render_button,
            restrict: {
                restriction: 'parent',
                elementRect: {top: 0, left: 0, bottom: 1, right: 1}
            },
        })
        // .resizable({
        //     // resize from all edges and corners
        //     edges: {left: true, right: true, bottom: true, top: true},
        //
        //     // keep the edges inside the parent
        //     restrictEdges: {
        //         outer: 'parent',
        //         endOnly: true,
        //     },
        //
        //     // minimum size
        //     restrictSize: {
        //         min: {width: 50, height: 50},
        //     },
        //
        //     inertia: true,
        // })
        // .on('resizemove', function (event) {
        //     var target = event.target,
        //         x = (parseFloat(target.getAttribute('data-x')) || 0),
        //         y = (parseFloat(target.getAttribute('data-y')) || 0);
        //
        //     // update the element's style
        //     target.style.width = event.rect.width + 'px';
        //     target.style.height = event.rect.height + 'px';
        //
        //     // translate when resizing from top or left edges
        //     x += event.deltaRect.left;
        //     y += event.deltaRect.top;
        //
        //     target.style.webkitTransform = target.style.transform =
        //         'translate(' + x + 'px,' + y + 'px)';
        //
        //     target.setAttribute('data-x', x);
        //     target.setAttribute('data-y', y);
        //     // target.textContent = Math.round(event.rect.width) + '\u00D7' + Math.round(event.rect.height);
        //     selectItem(event.target);
        // })
        .on('tap', function (event) {
            console.log('tap');
            var target = event.target;
            var size = parseInt(target.getAttribute('data-size'));
            var new_size = (size + 1) % 10;
            target.setAttribute('data-size', new_size);
            target.style.fontSize = new_size * 10 + 10;
            // $(event.currentTarget).remove();
            selectItem(event, event.target);
            render_button();
            event.preventDefault();
        })
        .on('hold', function (event) {
            console.log('hold');
            $(event.currentTarget).remove();
            render_button();
            event.preventDefault();
        });

    function selectItem(event, target, should_deselect) {
        event.stopPropagation();
        var hasClass = $(target).hasClass('selected');
        $(".resize-drag").removeClass("selected");
        $('#range-slider').attr('data-id', '');
        if (should_deselect && hasClass){
            // var image_style = $('#range-slider').attr('image-id');
            // image_style = image_style ? image_style : -1;
            // $('#range-slider').data(image_style);
            // $('.range-slider__value').text(image_style.toString());
        } else {
            $(target).addClass("selected");
            // event.currentTarget.classList.toggle('switch-bg');
            $('#range-slider').attr('data-id', target.id);
            var style = target.getAttribute('data-style');
            style = style ? style : -1;
            $('#range-slider').val(style);
            $('.range-slider__value').text(style.toString());
        }

        // render_button();
        // event.preventDefault();
    }

    $(".resize-drag").click(function (e) {
        $(".resize-drag").removeClass("selected");
        $(this).addClass("selected");
        e.stopPropagation();
    });

    function guidGenerator() {
        var S4 = function () {
            return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
        };
        return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
    }

    function stuff_add(evt) {
        // var selected_stuff = document.getElementById("stuff");
        evt.stopPropagation();
        var newContent = document.createTextNode(evt.currentTarget.textContent);
        var node = document.createElement("DIV");
        node.className = "resize-drag";
        node.id = guidGenerator();
        node.appendChild(newContent);
        node.setAttribute('data-size', 0);
        // node.style.fontSize = new_size * 10 + 10;
        document.getElementById("resize-container").appendChild(node);
    }

    function thing_add() {
        var selected_stuff = document.getElementById("thing");
        var newContent = document.createTextNode(selected_stuff[selected_stuff.selectedIndex].value);
        var node = document.createElement("DIV");
        node.className = "resize-drag";
        node.id = guidGenerator();
        node.appendChild(newContent);
        document.getElementById("resize-container").appendChild(node);
        node.setAttribute('data-size', 2);

    }

    function refresh_image(response) {
        response = JSON.parse(response);
        document.getElementById("img_pred").src = response.img_pred;
        document.getElementById("layout_pred").src = response.layout_pred;
    }

    function addRow(obj, size, location, feature) {
      // Get a reference to the table
      let tableRef = document.getElementById('table').getElementsByTagName('tbody')[0];

      // Insert a row at the end of the table
      let newRow = tableRef.insertRow(-1);

      // Insert a cell in the row at index 0
      newRow.insertCell(0).appendChild(document.createTextNode(obj));
      newRow.insertCell(1).appendChild(document.createTextNode(size + ''));
      newRow.insertCell(2).appendChild(document.createTextNode(location + ''));
      newRow.insertCell(3).appendChild(document.createTextNode(feature + ''));
    }

    function render_button() {
        console.log('render');
        var allObjects = [];
        $("tbody").children().remove();
        var container = document.getElementById("resize-container");
        var container_rect = interact.getElementRect(container);
        var containerOffsetLeft = container_rect.left;
        var containerOffsetTop = container_rect.top;
        var containerWidth = container_rect.width;
        var containerHeight = container_rect.height;
        var children = document.getElementsByClassName('resize-drag');
        if (children.length < 3) {
            return;
        }
        for (var i = 0; i < children.length; i++) {
            var rect = interact.getElementRect(children[i]);
            var height = rect.height / containerHeight;
            var width = rect.width / containerWidth;
            var left = (rect.left - containerOffsetLeft) / containerWidth;
            var top = (rect.top - containerOffsetTop) / containerHeight;
            var sx0 = left;
            var sy0 = top;
            var sx1 = width + left;
            var sy1 = height + sy0;
            var mean_x_s = 0.5 * (sx0 + sx1);
            var mean_y_s = 0.5 * (sy0 + sy1);
            var grid = 25 / 5;
            var location = Math.round(mean_x_s * (grid - 1)) + grid * Math.round(mean_y_s * (grid - 1));
            var size = parseInt(children[i].getAttribute('data-size'));
            var text = children[i].innerText;
            var style = children[i].getAttribute('data-style') ?
                parseInt(children[i].getAttribute('data-style')) :
                -1;
            allObjects.push({
                'height': height,
                'width': width,
                'left': left,
                'top': top,
                'text': text,
                'feature': style,
                'size': size,
                'location': location,
            });
            console.log(size, location, text);
            addRow(text, size, location, style);
        }
        console.log(allObjects);
        var image_id = document.getElementById('range-slider').getAttribute('image-id');
        var image_feature = image_id ? parseInt(image_id) : -1;
        addRow('background', '-', '-', image_feature);
        var results = {'image_id': image_feature, 'objects': allObjects};
        var url = 'get_data?data=' + JSON.stringify(results);
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function () {
            if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
                refresh_image(xmlHttp.responseText);
        };
        xmlHttp.open("GET", url, false); // true for asynchronous
        xmlHttp.send(null);
    }

    // document.getElementById("stuff-add").addEventListener("click", stuff_add);
    // document.getElementById("thing-add").addEventListener("click", thing_add);
    // document.getElementById("render-button").addEventListener("click", render_button);
    document.querySelectorAll("ul.drop-menu > li").forEach(function(e){e.addEventListener("click", stuff_add)})
    $(window).click(function(devt) {
        if (!devt.target.getAttribute('data-size') && !devt.target.getAttribute('max')){
            $(".resize-drag").removeClass("selected");
            var image_style = $('#range-slider').attr('image-id');
            image_style = image_style ? parseInt(image_style) : -1;
            $('#range-slider').val(image_style);
            $('.range-slider__value').text(image_style.toString());
            $('#range-slider').attr('data-id', '');
        }
    });
}

