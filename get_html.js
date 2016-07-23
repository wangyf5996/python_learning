var page = require('webpage').create(),
    system = require('system'),
    address;

if (system.args.length === 1) {
    phantom.exit(1);
} else {
    address = system.args[1];
    console.log(address);
    page.open(address, function (status) {
        if (status !== 'success') {
            console.log('fail');
            phantom.exit();
        } else {
            console.log('success')
            var sc = page.evaluate(function () {
                return document.body.innerHTML;
            });
            window.setTimeout(function () {
                console.log(sc);
                phantom.exit();
            }, 1000);
        }
    });
};